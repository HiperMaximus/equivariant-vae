# Copyright 2026 HiperMaximus
"""Generate the ordered UBC-OCEAN masked-holdout atlas and patch dataset.

On Kaggle, the script installs ``libvips`` and ``pyvips`` when they are absent.
The atlas is generated once and can then drive five independent extraction runs.
Each run owns complete WSIs, so an interrupted run can be repeated without
rerunning Otsu or touching the parts that already finished. A final merge only
streams the ordered part files together; it does not decode the WSIs again.

Typical Kaggle commands::

    python generate_ubc_ocean_test.py --stage atlas
    python generate_ubc_ocean_test.py --stage dataset --part 1
    # Repeat the previous command for parts 2, 3, 4, and 5 in other runs.
    python generate_ubc_ocean_test.py --stage merge --parts-dir ./all_parts
"""

# ruff: noqa: C901, PLR0913, PLR0914, PLR0915, T201

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import subprocess  # noqa: S404
import sys
import traceback
import zlib
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

# These values preserve the train/validation data contract. Changing them would
# make the test patches incomparable with the data seen by the two VAEs.
PATCH_SIZE = 256
CHANNELS = 3
TISSUE_FRACTION = 0.6
# Otsu already establishes that a patch contains tissue. This minimum applies
# only when incomplete mask paint is the sole reason to extract the patch: tiny
# annotation-edge specks would not provide a meaningful whole-patch example.
MIN_MASK_ONLY_FRACTION = 0.10
DEFAULT_PART_COUNT = 5
PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE

# Kaggle defines mask colors in RGB, not OpenCV's BGR order. Black is not a
# fourth tissue class: it means that the pathologist did not annotate that pixel.
MASK_CLASSES = ("tumor", "stroma", "necrosis")
SELECTION_SOURCES = frozenset({"mask", "otsu", "mask+otsu"})
MASK_STATUSES = frozenset({"annotated", "unannotated"})
MASK_LABELS = frozenset({*MASK_CLASSES, "ambiguous", "unknown"})

# The cohort is locked so a changed mask mount cannot silently create a
# different test set after results have already been inspected.
EXPECTED_WSI_COUNT = 152
LABEL_MAP = {"CC": 0, "EC": 1, "HGSC": 2, "LGSC": 3, "MC": 4}
EXPECTED_LABEL_COUNTS = {0: 33, 1: 36, 2: 56, 3: 15, 4: 12}
EXPECTED_WSI_LABEL_SHA256 = (
    "5c78418d836c8b31536ca72aa236b9d119415f99fc197856b00a6c9d356cf5e7"
)

# The evaluator already understands this 64-byte header and CHW byte layout.
# Reusing it lets each part, as well as the final merge, open with PatchShard.
HEADER_FORMAT = "<8sIQiiii3s25x"
HEADER_SIZE = 64
MAGIC = b"UBC_DATA"
VERSION = 1
LAYOUT = b"CHW"

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
COMPETITION_NAME = "UBC-OCEAN"
MASK_DATASET_OWNER = "sohier"
MASK_DATASET_NAME = "ubc-ovarian-cancer-competition-supplemental-masks"
# These legacy paths are still useful in error messages and for older Kaggle
# images. New workers can instead mount sources under owner/version folders, so
# omitted CLI paths are resolved below rather than blindly assuming these two.
DEFAULT_COMPETITION_ROOT = KAGGLE_INPUT_ROOT / COMPETITION_NAME
DEFAULT_MASK_DIR = KAGGLE_INPUT_ROOT / MASK_DATASET_NAME
KAGGLE_WORKING_DIR = Path("/kaggle/working")
DEFAULT_OUTPUT_DIR = KAGGLE_WORKING_DIR / "dataset"
PYVIPS_VERSION = "3.1.0"
APT_GET = "/usr/bin/apt-get"
ATLAS_NAME = "ubc_ocean_test_atlas.csv"
ATLAS_CHECKPOINT_NAME = "ubc_ocean_test_atlas_checkpoint.csv"
ATLAS_CHECKPOINT_STATE_NAME = "ubc_ocean_test_atlas_checkpoint.json"
ATLAS_INCOMPLETE_NAME = "ubc_ocean_test_atlas_incomplete.json"
ATLAS_CHECKPOINT_SCHEMA = "spec0017.atlas_checkpoint.v2"
BIN_NAME = "ubc_ocean_test.bin"
CSV_NAME = "ubc_ocean_test.csv"
PROVENANCE_NAME = "ubc_ocean_test_provenance.json"
COPY_CHUNK_BYTES = 8 * 1024 * 1024
SHA256_HEX_LENGTH = 64
ATLAS_COLUMNS = (
    "wsi_id",
    "label",
    "x",
    "y",
    "selection_source",
    "mask_status",
    "mask_label",
    "annotated_fraction",
    "tumor_fraction",
    "stroma_fraction",
    "necrosis_fraction",
)
PATCH_CSV_COLUMNS = ("idx", *ATLAS_COLUMNS)

# The Kaggle kernel build copies this readable file byte-for-byte to ``run.py``.
# Keeping the marker here lets the guarded push path recognize that generated
# upload without maintaining a second implementation.
KAGGLE_UBC_OCEAN_TEST_ATLAS_READY = True

if struct.calcsize(HEADER_FORMAT) != HEADER_SIZE:
    message = "Test patch header must stay exactly 64 bytes"
    raise RuntimeError(message)


@dataclass(frozen=True)
class HoldoutSlide:
    """One mask-selected non-TMA WSI and its numeric diagnosis label."""

    wsi_id: int
    label: int


@dataclass(frozen=True)
class AtlasRow:
    """One ordered patch coordinate with diagnosis and tissue annotations."""

    wsi_id: int
    label: int
    x: int
    y: int
    selection_source: str
    mask_status: str
    mask_label: str
    annotated_fraction: float
    tumor_fraction: float
    stroma_fraction: float
    necrosis_fraction: float


@dataclass(frozen=True)
class PartPlan:
    """One contiguous WSI range that can be generated in one Kaggle run."""

    number: int
    total_parts: int
    wsi_ids: tuple[int, ...]
    patch_count: int

    @property
    def stem(self) -> str:
        """Shared filename stem for this part's three artifacts."""
        return f"ubc_ocean_test_part_{self.number:02d}_of_{self.total_parts:02d}"


class _Digest(Protocol):
    """Small structural type shared by hashlib digest implementations."""

    def update(self, data: bytes) -> None:
        """Add bytes to the running digest."""

    def hexdigest(self) -> str:
        """Return the current lowercase hexadecimal digest."""


def resolve_competition_root(
    requested: Path | None,
    *,
    require_atlas_inputs: bool = True,
) -> Path:
    """Find the attached UBC-OCEAN files without guessing Kaggle's mount style.

    Kaggle historically mounted competitions directly below ``/kaggle/input``.
    Current workers may add a ``competitions`` folder. We accept only those two
    known identities. Atlas generation requires labels, WSIs, and thumbnails;
    dataset extraction deliberately requires only WSIs because the saved atlas
    already contains every label and selected coordinate.
    A user-supplied path is never replaced by an automatic fallback: a typo in
    an explicit resume command should fail instead of silently reading elsewhere.

    Returns:
        The verified directory containing the official competition files.

    """
    candidates = (
        (requested,)
        if requested is not None
        else (
            KAGGLE_INPUT_ROOT / COMPETITION_NAME,
            KAGGLE_INPUT_ROOT / "competitions" / COMPETITION_NAME,
        )
    )
    required = (
        ("train.csv", "train_images", "train_thumbnails")
        if require_atlas_inputs
        else ("train_images",)
    )
    return _first_complete_input_root(
        description="UBC-OCEAN competition",
        candidates=candidates,
        required_names=required,
    )


def resolve_mask_dir(requested: Path | None) -> Path:
    """Find the attached supplemental masks across Kaggle mount conventions.

    The old train/validation notebook used the direct slug path. Kaggle's newer
    source mounts preserve the owner and sometimes the selected version number.
    Checking this short, exact candidate list is safer and easier to understand
    than recursively accepting any directory that happens to contain PNG files.

    Returns:
        The verified directory containing all 152 numeric mask PNGs.

    Raises:
        FileNotFoundError: If no exact known mount contains all expected masks.

    """
    if requested is not None:
        candidates = (requested,)
    else:
        dataset_root = (
            KAGGLE_INPUT_ROOT / "datasets" / MASK_DATASET_OWNER / MASK_DATASET_NAME
        )
        candidates = (
            KAGGLE_INPUT_ROOT / MASK_DATASET_NAME,
            KAGGLE_INPUT_ROOT / MASK_DATASET_OWNER / MASK_DATASET_NAME,
            dataset_root,
            *_version_directories(dataset_root / "versions"),
        )

    searched: list[str] = []
    for candidate in candidates:
        searched.append(str(candidate))
        if _numeric_png_count(candidate) == EXPECTED_WSI_COUNT:
            return candidate
    message = (
        f"Could not resolve the supplemental mask directory with "
        f"{EXPECTED_WSI_COUNT} numeric PNG files. Searched: {', '.join(searched)}"
    )
    raise FileNotFoundError(message)


def resolve_atlas_checkpoint_dir(requested: Path | None) -> Path | None:
    """Find one attached atlas checkpoint, or honor one explicit directory.

    Kaggle mounts a previous kernel output somewhere below ``/kaggle/input``.
    Its exact parent path can vary, but our two checkpoint filenames do not.
    Automatic discovery is therefore safe only when exactly one complete pair
    exists; ambiguity or a half-pair fails instead of choosing silently.

    Returns:
        The directory containing one complete checkpoint pair, or ``None``.

    Raises:
        FileNotFoundError: If an explicit directory lacks either file, or an
            automatically discovered directory contains only half the pair.
        ValueError: If multiple attached checkpoint pairs are present.

    """
    if requested is not None:
        candidate_dirs = {requested}
    elif KAGGLE_INPUT_ROOT.is_dir():
        candidate_dirs = {
            path.parent
            for filename in (ATLAS_CHECKPOINT_NAME, ATLAS_CHECKPOINT_STATE_NAME)
            for path in KAGGLE_INPUT_ROOT.rglob(filename)
        }
    else:
        candidate_dirs = set()

    complete: list[Path] = []
    for candidate in sorted(candidate_dirs):
        checkpoint = candidate / ATLAS_CHECKPOINT_NAME
        state = candidate / ATLAS_CHECKPOINT_STATE_NAME
        if checkpoint.is_file() and state.is_file():
            complete.append(candidate)
            continue
        message = (
            "Atlas checkpoint directory must contain both the CSV and JSON state: "
            f"{candidate}"
        )
        raise FileNotFoundError(message)
    if len(complete) > 1:
        message = f"Found multiple attached atlas checkpoints: {complete}"
        raise ValueError(message)
    return complete[0] if complete else None


def _first_complete_input_root(
    *,
    description: str,
    candidates: Sequence[Path],
    required_names: Sequence[str],
) -> Path:
    """Return the first candidate with every required file or directory.

    Returns:
        The first complete input root in candidate order.

    Raises:
        FileNotFoundError: If none of the exact candidates is complete.

    """
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / required_name).exists() for required_name in required_names
        ):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    required = ", ".join(required_names)
    message = f"Could not resolve {description}; need {required}. Searched: {searched}"
    raise FileNotFoundError(message)


def _version_directories(versions_root: Path) -> tuple[Path, ...]:
    """List only mounted numeric dataset versions, newest first.

    Returns:
        Existing numeric version directories in descending numeric order.

    """
    if not versions_root.is_dir():
        return ()
    return tuple(
        sorted(
            (
                child
                for child in versions_root.iterdir()
                if child.is_dir() and child.name.isdigit()
            ),
            key=lambda child: int(child.name),
            reverse=True,
        ),
    )


def _numeric_png_count(directory: Path) -> int:
    """Count mask-like files without reading their multi-megabyte pixel data.

    Returns:
        The number of top-level PNGs whose stems are numeric WSI IDs.

    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png" and path.stem.isdigit()
    )


def load_holdout_slides(
    labels_csv: Path,
    mask_dir: Path,
    *,
    expected_wsi_count: int = EXPECTED_WSI_COUNT,
) -> list[HoldoutSlide]:
    """Select mask-bearing non-TMA slides from official UBC metadata.

    Returns:
        Numerically ordered held-out slides with mapped labels.

    Raises:
        ValueError: If masks and official non-TMA metadata do not match.

    """
    # A mask filename means "reserve this WSI for testing". The atlas stage later
    # reads its painted pixels as incomplete annotations, but filename membership
    # is still what defines the held-out WSI cohort.
    mask_ids = {
        int(path.stem)
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    if len(mask_ids) != expected_wsi_count:
        message = (
            f"Expected {expected_wsi_count} mask IDs, found {len(mask_ids)} in "
            f"{mask_dir}"
        )
        raise ValueError(message)

    selected: dict[int, HoldoutSlide] = {}
    with labels_csv.open(newline="", encoding="utf-8") as labels_file:
        for row in csv.DictReader(labels_file):
            wsi_id = int(_required(row, "image_id"))
            if wsi_id not in mask_ids:
                continue
            if _parse_bool(_required(row, "is_tma")):
                message = f"Masked holdout WSI {wsi_id} is marked as TMA"
                raise ValueError(message)
            label_name = _required(row, "label")
            if label_name not in LABEL_MAP:
                message = f"Unknown label {label_name!r} for WSI {wsi_id}"
                raise ValueError(message)
            if wsi_id in selected:
                message = f"Duplicate train.csv row for masked WSI {wsi_id}"
                raise ValueError(message)
            selected[wsi_id] = HoldoutSlide(wsi_id, LABEL_MAP[label_name])

    missing_ids = sorted(mask_ids - selected.keys())
    if missing_ids:
        message = f"Mask IDs missing from non-TMA train.csv rows: {missing_ids}"
        raise ValueError(message)
    if expected_wsi_count == EXPECTED_WSI_COUNT:
        observed_label_counts = {
            label: sum(slide.label == label for slide in selected.values())
            for label in LABEL_MAP.values()
        }
        if observed_label_counts != EXPECTED_LABEL_COUNTS:
            message = (
                f"Masked holdout label counts differ from the canonical split: "
                f"{observed_label_counts}"
            )
            raise ValueError(message)
        selected_labels = {wsi_id: slide.label for wsi_id, slide in selected.items()}
        if _wsi_label_digest(selected_labels) != EXPECTED_WSI_LABEL_SHA256:
            message = "Mask-selected WSI IDs/labels differ from the canonical holdout"
            raise ValueError(message)
    return [selected[wsi_id] for wsi_id in sorted(selected)]


def iter_tissue_coordinates(
    tissue_mask: NDArray[np.uint8],
    *,
    wsi_width: int,
    wsi_height: int,
    patch_size: int = PATCH_SIZE,
    tissue_fraction: float = TISSUE_FRACTION,
) -> Iterator[tuple[int, int]]:
    """Yield the historical Otsu-selected grid coordinates in ``y,x`` order.

    Yields:
        Full-patch ``(x, y)`` coordinates passing the projected tissue threshold.

    """
    thumb_height, thumb_width = tissue_mask.shape
    # The thumbnail is a smaller view of the same slide. These two scale factors
    # map every full-resolution patch rectangle onto the corresponding mask
    # rectangle without ever loading the full WSI into NumPy memory.
    scale_x = wsi_width / thumb_width
    scale_y = wsi_height / thumb_height
    projected_patch_area = (patch_size / scale_x) * (patch_size / scale_y)
    tissue_pixel_threshold = projected_patch_area * tissue_fraction

    # Y is the outer loop because PNGs decode top-to-bottom. Within one row we
    # move left-to-right. The resulting wsi_id,y,x order is both deterministic
    # for restart/merge and friendly to libvips' sequential reader.
    for y in range(0, wsi_height - patch_size + 1, patch_size):
        for x in range(0, wsi_width - patch_size + 1, patch_size):
            # ``int`` deliberately preserves the floor-based projection used by
            # the historical generator; changing rounding would change the
            # selected test patches at thumbnail-cell boundaries.
            tx_start = int(x / scale_x)
            ty_start = int(y / scale_y)
            tx_end = int((x + patch_size) / scale_x)
            ty_end = int((y + patch_size) / scale_y)
            mask_region = tissue_mask[ty_start:ty_end, tx_start:tx_end]
            # Strictly greater than 60%, rather than greater-or-equal, is also
            # inherited from train/validation and is tested explicitly.
            if np.count_nonzero(mask_region) > tissue_pixel_threshold:
                yield x, y


def mask_strip_fractions(
    blob: bytes,
    *,
    grid_width: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Measure annotation and class fractions for one patch-high mask row.

    The real masks contain mostly canonical colors plus a small number of
    antialiased edge colors. A unique strongest RGB channel supplies one class.
    When channels tie for the maximum, the pixel contributes to every tied class;
    its annotation coverage is still counted only once. This preserves the user's
    chosen overlap convention without treating every faint nonmaximum channel in
    an antialiased edge as another class.

    Returns:
        Annotated, tumor, stroma, and necrosis fraction arrays, one value per
        horizontal patch.

    Raises:
        ValueError: If bytes have the wrong shape.

    """
    if grid_width <= 0 or grid_width % PATCH_SIZE != 0:
        message = (
            f"Mask strip width must be a positive multiple of {PATCH_SIZE}, "
            f"got {grid_width}"
        )
        raise ValueError(message)
    expected_values = PATCH_SIZE * grid_width * CHANNELS
    pixels = np.frombuffer(blob, dtype=np.uint8)
    if pixels.size != expected_values:
        message = (
            f"Mask strip returned {pixels.size} values; expected {expected_values}"
        )
        raise ValueError(message)
    rgb = pixels.reshape(PATCH_SIZE, grid_width, CHANNELS)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    nonblack = (red > 0) | (green > 0) | (blue > 0)
    tumor = nonblack & (red >= green) & (red >= blue)
    stroma = nonblack & (green >= red) & (green >= blue)
    necrosis = nonblack & (blue >= red) & (blue >= green)

    grid_columns = grid_width // PATCH_SIZE

    def fractions(class_pixels: NDArray[np.bool_]) -> NDArray[np.float64]:
        counts = class_pixels.reshape(PATCH_SIZE, grid_columns, PATCH_SIZE).sum(
            axis=(0, 2),
        )
        return counts.astype(np.float64) / PATCH_PIXELS

    return fractions(nonblack), fractions(tumor), fractions(stroma), fractions(necrosis)


def _mask_label(tumor: float, stroma: float, necrosis: float) -> str:
    fractions = (tumor, stroma, necrosis)
    largest = max(fractions)
    if largest <= 0.0:
        return "unknown"
    winners = [
        name
        for name, fraction in zip(
            MASK_CLASSES,
            fractions,
            strict=True,
        )
        if fraction == largest
    ]
    return winners[0] if len(winners) == 1 else "ambiguous"


def _atlas_row_dict(row: AtlasRow, *, idx: int | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "wsi_id": row.wsi_id,
        "label": row.label,
        "x": row.x,
        "y": row.y,
        "selection_source": row.selection_source,
        "mask_status": row.mask_status,
        "mask_label": row.mask_label,
        "annotated_fraction": row.annotated_fraction,
        "tumor_fraction": row.tumor_fraction,
        "stroma_fraction": row.stroma_fraction,
        "necrosis_fraction": row.necrosis_fraction,
    }
    if idx is not None:
        values["idx"] = idx
    return values


def _update_atlas_rows_digest(digest: _Digest, row: AtlasRow) -> None:
    """Bind one logical atlas row independent of CSV newline conventions."""
    canonical = json.dumps(
        _atlas_row_dict(row),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest.update(canonical.encode("ascii"))
    digest.update(b"\n")


def _parse_atlas_row(raw_row: dict[str, str | None]) -> AtlasRow:
    """Parse and cross-check one persisted atlas or part-CSV row.

    Returns:
        A fully validated atlas row.

    Raises:
        ValueError: If annotation fields are missing or internally inconsistent.

    """
    row = AtlasRow(
        wsi_id=int(_required(raw_row, "wsi_id")),
        label=int(_required(raw_row, "label")),
        x=int(_required(raw_row, "x")),
        y=int(_required(raw_row, "y")),
        selection_source=_required(raw_row, "selection_source"),
        mask_status=_required(raw_row, "mask_status"),
        mask_label=_required(raw_row, "mask_label"),
        annotated_fraction=float(_required(raw_row, "annotated_fraction")),
        tumor_fraction=float(_required(raw_row, "tumor_fraction")),
        stroma_fraction=float(_required(raw_row, "stroma_fraction")),
        necrosis_fraction=float(_required(raw_row, "necrosis_fraction")),
    )
    if row.selection_source not in SELECTION_SOURCES:
        message = f"Invalid atlas selection source {row.selection_source!r}"
        raise ValueError(message)
    if row.mask_status not in MASK_STATUSES:
        message = f"Invalid atlas mask status {row.mask_status!r}"
        raise ValueError(message)
    if row.mask_label not in MASK_LABELS:
        message = f"Invalid atlas mask label {row.mask_label!r}"
        raise ValueError(message)

    fractions = (
        row.annotated_fraction,
        row.tumor_fraction,
        row.stroma_fraction,
        row.necrosis_fraction,
    )
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
        message = f"Atlas mask fractions must be finite and within [0, 1]: {fractions}"
        raise ValueError(message)
    class_fractions = fractions[1:]
    if any(
        class_fraction > row.annotated_fraction + 1e-12
        for class_fraction in class_fractions
    ):
        message = (
            "Atlas class fractions cannot exceed annotated_fraction individually: "
            f"{fractions}"
        )
        raise ValueError(message)
    class_sum = sum(class_fractions)
    if class_sum + 1e-12 < row.annotated_fraction:
        message = (
            "Atlas class fractions must cover every annotated pixel: "
            f"{row.annotated_fraction} > {class_sum}"
        )
        raise ValueError(message)

    expected_label = _mask_label(
        row.tumor_fraction,
        row.stroma_fraction,
        row.necrosis_fraction,
    )
    has_annotation = row.annotated_fraction > 0.0
    if not has_annotation:
        expected_status = "unannotated"
        allowed_sources = {"otsu"}
    else:
        expected_status = "annotated"
        allowed_sources = {"mask", "mask+otsu"}
    if row.mask_status != expected_status or row.mask_label != expected_label:
        message = (
            "Atlas mask status/label disagree with its fractions: "
            f"{row.mask_status!r}, {row.mask_label!r}, {fractions}"
        )
        raise ValueError(message)
    if row.selection_source not in allowed_sources:
        message = (
            f"Atlas selection source {row.selection_source!r} disagrees with "
            f"annotated_fraction={row.annotated_fraction}"
        )
        raise ValueError(message)
    return row


def _atlas_checkpoint_paths(atlas_path: Path) -> tuple[Path, Path]:
    """Return the stable CSV/state pair used between complete WSI boundaries.

    Returns:
        Checkpoint CSV and JSON state paths beside the final atlas.

    """
    return (
        atlas_path.parent / ATLAS_CHECKPOINT_NAME,
        atlas_path.parent / ATLAS_CHECKPOINT_STATE_NAME,
    )


def _write_atlas_checkpoint_state(
    *,
    state_path: Path,
    completed_slides: Sequence[HoldoutSlide],
    patch_count: int,
    csv_bytes: int,
    rows_sha256: str,
    expected_wsi_count: int,
) -> None:
    """Atomically commit the byte boundary of the last complete WSI."""
    payload = {
        "schema_version": ATLAS_CHECKPOINT_SCHEMA,
        "atlas_contract": "spec0017.masked_holdout_test.v4",
        "completed_wsi_ids": [slide.wsi_id for slide in completed_slides],
        "completed_wsi_count": len(completed_slides),
        "expected_wsi_count": expected_wsi_count,
        "patch_count": patch_count,
        "csv_bytes": csv_bytes,
        "rows_sha256": rows_sha256,
        "ordered_by": ["wsi_id", "y", "x"],
    }
    temporary = state_path.with_suffix(f"{state_path.suffix}.partial")
    with temporary.open("w", encoding="utf-8") as state_file:
        json.dump(payload, state_file, indent=2, sort_keys=True)
        state_file.write("\n")
        state_file.flush()
        os.fsync(state_file.fileno())
    temporary.replace(state_path)


def _publish_atlas_checkpoint_after_error(
    *,
    atlas_path: Path,
    slides: Sequence[HoldoutSlide],
    error: Exception,
) -> bool:
    """Mark a useful partial atlas as an intentionally publishable output.

    Kaggle exposes ``/kaggle/working`` files from a completed background run,
    while the failed version-2 run exposed no output files. Returning success
    after a Python-level atlas error is therefore deliberate *only* when at
    least one whole WSI has already been committed. The missing final atlas and
    this explicit marker prevent the checkpoint-only result from being mistaken
    for a finished dataset.

    Returns:
        True when a valid nonempty checkpoint was marked for publication.

    """
    if atlas_path.exists():
        return False
    try:
        completed_count, _patch_count, _digest = _resume_atlas_checkpoint(
            atlas_path=atlas_path,
            slides=slides,
            resume_checkpoint_dir=None,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if completed_count <= 0:
        return False

    checkpoint_path, state_path = _atlas_checkpoint_paths(atlas_path)

    marker_path = atlas_path.parent / ATLAS_INCOMPLETE_NAME
    marker = {
        "status": "incomplete_checkpoint_only",
        "completed_wsi_count": completed_count,
        "checkpoint_csv": checkpoint_path.name,
        "checkpoint_state": state_path.name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "resume_instruction": (
            "Attach this output to a new atlas run; the generator discovers the "
            "single checkpoint pair automatically."
        ),
    }
    temporary = marker_path.with_suffix(f"{marker_path.suffix}.partial")
    with temporary.open("w", encoding="utf-8") as marker_file:
        json.dump(marker, marker_file, indent=2, sort_keys=True)
        marker_file.write("\n")
        marker_file.flush()
        os.fsync(marker_file.fileno())
    temporary.replace(marker_path)
    return True


def _resume_atlas_checkpoint(  # noqa: PLR0912
    *,
    atlas_path: Path,
    slides: Sequence[HoldoutSlide],
    resume_checkpoint_dir: Path | None,
) -> tuple[int, int, _Digest]:
    """Restore and validate the last committed whole-WSI atlas prefix.

    A crash may append part of the next WSI to the CSV before the JSON state is
    advanced. The stored byte boundary is therefore authoritative: resume first
    truncates that uncommitted tail, then validates every retained row and the
    exact ordered WSI prefix. An attached read-only checkpoint directory is copied
    once into the writable output directory before this validation.

    Returns:
        Completed WSI count, retained atlas patch count, and the validated
        logical-row digest ready to continue.

    Raises:
        FileNotFoundError: If only half of a requested checkpoint pair exists.
        TypeError: If checkpoint state is not a JSON object.
        ValueError: If checkpoint state or retained atlas rows are inconsistent.

    """
    checkpoint_path, state_path = _atlas_checkpoint_paths(atlas_path)
    if resume_checkpoint_dir is not None:
        source_checkpoint = resume_checkpoint_dir / ATLAS_CHECKPOINT_NAME
        source_state = resume_checkpoint_dir / ATLAS_CHECKPOINT_STATE_NAME
        if not source_checkpoint.is_file() or not source_state.is_file():
            message = (
                "Resume directory must contain both atlas checkpoint files: "
                f"{source_checkpoint}, {source_state}"
            )
            raise FileNotFoundError(message)
        if source_checkpoint.resolve() != checkpoint_path.resolve():
            shutil.copyfile(source_checkpoint, checkpoint_path)
            shutil.copyfile(source_state, state_path)

    checkpoint_exists = checkpoint_path.exists()
    state_exists = state_path.exists()
    if not checkpoint_exists and not state_exists:
        return 0, 0, hashlib.sha256()
    if checkpoint_exists != state_exists:
        message = (
            "Atlas resume requires both checkpoint CSV and JSON state; found "
            f"csv={checkpoint_exists}, state={state_exists}"
        )
        raise FileNotFoundError(message)

    raw_state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw_state, dict):
        message = "Atlas checkpoint state must be a JSON object"
        raise TypeError(message)
    state = cast("dict[str, object]", raw_state)
    if state.get("schema_version") != ATLAS_CHECKPOINT_SCHEMA:
        message = f"Unsupported atlas checkpoint schema {state.get('schema_version')!r}"
        raise ValueError(message)
    if state.get("atlas_contract") != "spec0017.masked_holdout_test.v4":
        message = f"Atlas checkpoint contract mismatch: {state.get('atlas_contract')!r}"
        raise ValueError(message)

    completed_wsi_ids = state.get("completed_wsi_ids")
    completed_wsi_count = state.get("completed_wsi_count")
    patch_count = state.get("patch_count")
    csv_bytes = state.get("csv_bytes")
    expected_rows_sha256 = state.get("rows_sha256")
    expected_wsi_count = state.get("expected_wsi_count")
    integer_fields = (completed_wsi_count, patch_count, csv_bytes, expected_wsi_count)
    if (
        not isinstance(completed_wsi_ids, list)
        or any(not isinstance(value, int) for value in completed_wsi_ids)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields
        )
        or not isinstance(expected_rows_sha256, str)
        or len(expected_rows_sha256) != SHA256_HEX_LENGTH
    ):
        message = "Atlas checkpoint state has invalid count or WSI-ID fields"
        raise ValueError(message)
    completed_count = cast("int", completed_wsi_count)
    retained_patch_count = cast("int", patch_count)
    stable_csv_bytes = cast("int", csv_bytes)
    invalid_progress = (
        completed_count != len(completed_wsi_ids)
        or completed_count < 0
        or completed_count > len(slides)
    )
    invalid_sizes = retained_patch_count < 0 or stable_csv_bytes <= 0
    if invalid_progress or invalid_sizes or expected_wsi_count != len(slides):
        message = "Atlas checkpoint counts do not match the requested cohort"
        raise ValueError(message)
    expected_ids = [slide.wsi_id for slide in slides[:completed_count]]
    if completed_wsi_ids != expected_ids:
        message = (
            "Atlas checkpoint WSI IDs are not the exact ordered cohort prefix: "
            f"{completed_wsi_ids!r} != {expected_ids!r}"
        )
        raise ValueError(message)
    if checkpoint_path.stat().st_size < stable_csv_bytes:
        message = "Atlas checkpoint CSV is shorter than its committed byte boundary"
        raise ValueError(message)
    with checkpoint_path.open("r+b") as checkpoint_file:
        checkpoint_file.truncate(stable_csv_bytes)

    expected_labels = {slide.wsi_id: slide.label for slide in slides[:completed_count]}
    rows_digest = hashlib.sha256()
    actual_ids: list[int] = []
    actual_row_count = 0
    for row in iter_atlas_rows(checkpoint_path):
        actual_row_count += 1
        if not actual_ids or actual_ids[-1] != row.wsi_id:
            actual_ids.append(row.wsi_id)
        if (
            row.wsi_id not in expected_labels
            or row.label != expected_labels[row.wsi_id]
        ):
            message = "Atlas checkpoint row labels do not match the held-out cohort"
            raise ValueError(message)
        _update_atlas_rows_digest(rows_digest, row)
    if actual_row_count != retained_patch_count:
        message = (
            f"Atlas checkpoint row count {actual_row_count} != state "
            f"{retained_patch_count}"
        )
        raise ValueError(message)
    if actual_ids != expected_ids:
        message = (
            f"Atlas checkpoint rows cover {actual_ids!r}, expected {expected_ids!r}"
        )
        raise ValueError(message)
    if rows_digest.hexdigest() != expected_rows_sha256:
        message = "Atlas checkpoint rows do not match their committed SHA-256"
        raise ValueError(message)
    return completed_count, retained_patch_count, rows_digest


def generate_atlas(  # noqa: PLR0912
    *,
    slides: Sequence[HoldoutSlide],
    thumbnail_dir: Path,
    wsi_dir: Path,
    mask_dir: Path,
    atlas_path: Path,
    resume_checkpoint_dir: Path | None = None,
    cv2_module: ModuleType | None = None,
    pyvips_module: ModuleType | None = None,
) -> int:
    """Write the downloadable ordered atlas with whole-WSI checkpoints.

    Returns:
        Number of atlas patch rows written.

    Raises:
        FileExistsError: If a final atlas already exists at the requested path.
        FileNotFoundError: If a requested resume checkpoint is incomplete or a
            selected WSI thumbnail cannot be read.
        ValueError: If a checkpoint is inconsistent or Otsu/mask selection
            produces no patches for a held-out WSI.

    """
    cv2 = _load_module("cv2") if cv2_module is None else cv2_module
    pyvips = _load_module("pyvips") if pyvips_module is None else pyvips_module
    _disable_vips_operation_cache(pyvips)
    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    if atlas_path.exists():
        message = f"Final atlas already exists; refusing to overwrite it: {atlas_path}"
        raise FileExistsError(message)
    checkpoint_path, checkpoint_state_path = _atlas_checkpoint_paths(atlas_path)
    completed_wsi_count, total_patches, rows_digest = _resume_atlas_checkpoint(
        atlas_path=atlas_path,
        slides=slides,
        resume_checkpoint_dir=resume_checkpoint_dir,
    )
    file_mode = "a" if completed_wsi_count else "w"
    with checkpoint_path.open(file_mode, newline="", encoding="utf-8") as atlas_file:
        writer = csv.DictWriter(
            atlas_file,
            fieldnames=ATLAS_COLUMNS,
        )
        if completed_wsi_count == 0:
            writer.writeheader()
            atlas_file.flush()
            os.fsync(atlas_file.fileno())
            _write_atlas_checkpoint_state(
                state_path=checkpoint_state_path,
                completed_slides=(),
                patch_count=0,
                csv_bytes=checkpoint_path.stat().st_size,
                rows_sha256=rows_digest.hexdigest(),
                expected_wsi_count=len(slides),
            )
        else:
            print(
                f"Resuming atlas after {completed_wsi_count}/{len(slides)} WSIs "
                f"and {total_patches} patches",
                flush=True,
            )

        for slide_number, slide_info in enumerate(slides, start=1):
            if slide_number <= completed_wsi_count:
                continue
            wsi_path = wsi_dir / f"{slide_info.wsi_id}.png"
            thumbnail_path = thumbnail_dir / f"{slide_info.wsi_id}_thumbnail.png"
            mask_path = mask_dir / f"{slide_info.wsi_id}.png"
            slide = pyvips.Image.new_from_file(str(wsi_path))
            mask = pyvips.Image.new_from_file(
                str(mask_path),
                access="sequential",
                fail=True,
            )
            if int(mask.bands) != CHANNELS:
                message = f"Mask {mask_path} has {mask.bands} channels; expected RGB"
                raise ValueError(message)
            if int(mask.width) != int(slide.width) or int(mask.height) != int(
                slide.height,
            ):
                message = (
                    f"Mask {mask_path} dimensions {mask.width}x{mask.height} do not "
                    f"match WSI {slide.width}x{slide.height}"
                )
                raise ValueError(message)
            thumbnail = cv2.imread(str(thumbnail_path), cv2.IMREAD_COLOR)
            if thumbnail is None:
                message = f"Could not read thumbnail {thumbnail_path}"
                raise FileNotFoundError(message)

            # Stained tissue is usually more saturated than the pale slide
            # background. Otsu chooses the saturation cutoff independently for
            # each thumbnail, avoiding a hand-tuned fixed colour threshold.
            hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
            _threshold, tissue_mask = cv2.threshold(
                hsv[:, :, 1],
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            otsu_coordinates = set(
                iter_tissue_coordinates(
                    cast("NDArray[np.uint8]", tissue_mask),
                    wsi_width=int(slide.width),
                    wsi_height=int(slide.height),
                ),
            )

            # One mask strip is only 256 pixels high. This keeps memory bounded
            # while replacing thousands of per-patch PNG calls with one
            # sequential read per atlas row.
            grid_width = int(slide.width) // PATCH_SIZE * PATCH_SIZE
            grid_height = int(slide.height) // PATCH_SIZE * PATCH_SIZE
            mask_region = pyvips.Region.new(mask)
            slide_patch_count = 0
            for y in range(0, grid_height, PATCH_SIZE):
                mask_blob = mask_region.fetch(0, y, grid_width, PATCH_SIZE)
                try:
                    annotated, tumor, stroma, necrosis = mask_strip_fractions(
                        mask_blob,
                        grid_width=grid_width,
                    )
                except ValueError as error:
                    message = f"Invalid mask colors in WSI {slide_info.wsi_id}, y={y}"
                    raise ValueError(message) from error

                for column, x in enumerate(range(0, grid_width, PATCH_SIZE)):
                    tumor_fraction = float(tumor[column])
                    stroma_fraction = float(stroma[column])
                    necrosis_fraction = float(necrosis[column])
                    annotated_fraction = float(annotated[column])
                    has_mask = annotated_fraction > 0.0
                    passes_otsu = (x, y) in otsu_coordinates
                    if not has_mask and not passes_otsu:
                        continue
                    source = (
                        "mask+otsu"
                        if has_mask and passes_otsu
                        else "mask"
                        if has_mask
                        else "otsu"
                    )
                    row = AtlasRow(
                        wsi_id=slide_info.wsi_id,
                        label=slide_info.label,
                        x=x,
                        y=y,
                        selection_source=source,
                        mask_status="annotated" if has_mask else "unannotated",
                        mask_label=_mask_label(
                            tumor_fraction,
                            stroma_fraction,
                            necrosis_fraction,
                        ),
                        annotated_fraction=annotated_fraction,
                        tumor_fraction=tumor_fraction,
                        stroma_fraction=stroma_fraction,
                        necrosis_fraction=necrosis_fraction,
                    )
                    writer.writerow(_atlas_row_dict(row))
                    _update_atlas_rows_digest(rows_digest, row)
                    slide_patch_count += 1
                del annotated, tumor, stroma, necrosis, mask_blob
            if slide_patch_count == 0:
                message = (
                    f"Mask and Otsu selected zero patches for WSI {slide_info.wsi_id}"
                )
                raise ValueError(message)
            total_patches += slide_patch_count
            # The CSV bytes become resumable only after the entire WSI is on
            # disk. If the next WSI is interrupted, its tail is truncated back
            # to this committed byte boundary on resume.
            atlas_file.flush()
            os.fsync(atlas_file.fileno())
            _write_atlas_checkpoint_state(
                state_path=checkpoint_state_path,
                completed_slides=slides[:slide_number],
                patch_count=total_patches,
                csv_bytes=checkpoint_path.stat().st_size,
                rows_sha256=rows_digest.hexdigest(),
                expected_wsi_count=len(slides),
            )
            print(
                f"Atlas {slide_number}/{len(slides)}: WSI {slide_info.wsi_id} "
                f"-> {slide_patch_count} patches",
                flush=True,
            )
            del mask_region, mask, slide

    checkpoint_path.replace(atlas_path)
    checkpoint_state_path.unlink(missing_ok=True)
    (atlas_path.parent / ATLAS_INCOMPLETE_NAME).unlink(missing_ok=True)
    print(f"Atlas saved to {atlas_path} ({total_patches} patches)", flush=True)
    return total_patches


def inspect_atlas(
    atlas_path: Path,
    *,
    expected_wsi_count: int = EXPECTED_WSI_COUNT,
) -> tuple[int, int]:
    """Return atlas WSI and extraction-eligible patch counts after validation.

    Returns:
        WSI count and extraction-eligible patch count.

    Raises:
        ValueError: If atlas membership, labels, or ordering are invalid.

    """
    candidate_patch_count = 0
    eligible_patch_count = 0
    labels_by_wsi: dict[int, int] = {}
    eligible_wsi_ids: set[int] = set()
    for row in iter_atlas_rows(atlas_path):
        candidate_patch_count += 1
        previous_label = labels_by_wsi.setdefault(row.wsi_id, row.label)
        if previous_label != row.label:
            message = f"Atlas has inconsistent labels for WSI {row.wsi_id}"
            raise ValueError(message)
        if _is_extraction_eligible(row):
            eligible_patch_count += 1
            eligible_wsi_ids.add(row.wsi_id)
    if len(labels_by_wsi) != expected_wsi_count:
        message = (
            f"Expected {expected_wsi_count} atlas WSIs, found {len(labels_by_wsi)}"
        )
        raise ValueError(message)
    all_wsi_ids = set(labels_by_wsi)
    if eligible_wsi_ids != all_wsi_ids:
        missing = sorted(all_wsi_ids - eligible_wsi_ids)
        message = f"Atlas WSIs have no extraction-eligible patches: {missing}"
        raise ValueError(message)
    if expected_wsi_count == EXPECTED_WSI_COUNT:
        observed_label_counts = {
            label: sum(value == label for value in labels_by_wsi.values())
            for label in LABEL_MAP.values()
        }
        if observed_label_counts != EXPECTED_LABEL_COUNTS:
            message = (
                "Atlas label counts differ from canonical split: "
                f"{observed_label_counts}"
            )
            raise ValueError(message)
        observed_digest = _wsi_label_digest(labels_by_wsi)
        if observed_digest != EXPECTED_WSI_LABEL_SHA256:
            message = "Atlas WSI IDs/labels differ from the canonical holdout"
            raise ValueError(message)
    projected_gib = (
        HEADER_SIZE + eligible_patch_count * CHANNELS * PATCH_SIZE**2
    ) / 2**30
    print(
        f"Atlas contains {candidate_patch_count} candidates from "
        f"{len(labels_by_wsi)} WSIs; {eligible_patch_count} pass extraction "
        f"eligibility and project to {projected_gib:.3f} GiB",
        flush=True,
    )
    return len(labels_by_wsi), eligible_patch_count


def plan_dataset_parts(
    atlas_path: Path,
    *,
    part_count: int = DEFAULT_PART_COUNT,
) -> list[PartPlan]:
    """Divide the ordered atlas into contiguous, approximately equal work.

    A WSI is the restart boundary: it belongs wholly to one part. This avoids
    reopening the same very large PNG in two Kaggle sessions. Patch counts are
    used instead of WSI counts because they are a better estimate of extraction
    time and output size.

    Returns:
        Ordered part plans whose concatenated WSI IDs cover the atlas exactly.

    Raises:
        ValueError: If the requested part count cannot make non-empty parts.

    """
    if part_count < 1:
        message = f"part_count must be positive, got {part_count}"
        raise ValueError(message)

    patches_by_wsi: list[tuple[int, int]] = []
    for wsi_id, rows in groupby(
        iter_extraction_rows(atlas_path),
        key=lambda row: row.wsi_id,
    ):
        patches_by_wsi.append((wsi_id, sum(1 for _row in rows)))
    if part_count > len(patches_by_wsi):
        message = (
            f"Cannot make {part_count} non-empty parts from {len(patches_by_wsi)} WSIs"
        )
        raise ValueError(message)

    plans: list[PartPlan] = []
    next_wsi = 0
    remaining_patches = sum(count for _wsi_id, count in patches_by_wsi)

    for number in range(1, part_count + 1):
        remaining_parts = part_count - number + 1
        target = remaining_patches / remaining_parts
        max_end = len(patches_by_wsi) - (remaining_parts - 1)
        selected: list[int] = []
        selected_patches = 0

        while next_wsi < max_end:
            wsi_id, patch_count = patches_by_wsi[next_wsi]
            # Once this part has a WSI, stop before the next one when doing so
            # is closer to the current equal-work target. This keeps ranges
            # contiguous while adapting to different amounts of tissue.
            if selected and abs(selected_patches - target) <= abs(
                selected_patches + patch_count - target,
            ):
                break
            selected.append(wsi_id)
            selected_patches += patch_count
            next_wsi += 1

        if not selected:
            wsi_id, patch_count = patches_by_wsi[next_wsi]
            selected.append(wsi_id)
            selected_patches = patch_count
            next_wsi += 1

        plans.append(
            PartPlan(
                number=number,
                total_parts=part_count,
                wsi_ids=tuple(selected),
                patch_count=selected_patches,
            ),
        )
        remaining_patches -= selected_patches

    return plans


def print_part_plan(plans: Sequence[PartPlan]) -> None:
    """Print the exact WSI range, patch count, and size of every future run."""
    for plan in plans:
        projected_gib = (
            HEADER_SIZE + plan.patch_count * CHANNELS * PATCH_SIZE**2
        ) / 2**30
        print(
            f"Part {plan.number}/{plan.total_parts}: WSIs {plan.wsi_ids[0]}.."
            f"{plan.wsi_ids[-1]} ({len(plan.wsi_ids)} WSIs), "
            f"{plan.patch_count} patches, {projected_gib:.3f} GiB",
            flush=True,
        )


def iter_atlas_rows(atlas_path: Path) -> Iterator[AtlasRow]:
    """Read atlas rows while enforcing the generation order and label contract.

    Yields:
        Parsed atlas rows in file order.

    Raises:
        ValueError: If a row has an invalid label or breaks strict ordering.

    """
    previous_key: tuple[int, int, int] | None = None
    with atlas_path.open(newline="", encoding="utf-8") as atlas_file:
        for raw_row in csv.DictReader(atlas_file):
            row = _parse_atlas_row(raw_row)
            if row.label not in LABEL_MAP.values():
                message = f"Invalid numeric label {row.label} for WSI {row.wsi_id}"
                raise ValueError(message)
            if row.x < 0 or row.y < 0:
                message = f"Atlas coordinates must be nonnegative: ({row.x}, {row.y})"
                raise ValueError(message)
            if row.x % PATCH_SIZE != 0 or row.y % PATCH_SIZE != 0:
                message = (
                    f"Atlas coordinates must align to the {PATCH_SIZE}-pixel grid: "
                    f"({row.x}, {row.y})"
                )
                raise ValueError(message)
            key = (row.wsi_id, row.y, row.x)
            if previous_key is not None and key <= previous_key:
                message = (
                    "Atlas rows must be unique and ordered by wsi_id,y,x; "
                    f"found {key} after {previous_key}"
                )
                raise ValueError(message)
            previous_key = key
            yield row


def _is_extraction_eligible(row: AtlasRow) -> bool:
    """Check whether one lossless atlas candidate belongs in patch binaries.

    Returns:
        True for Otsu patches or mask-only patches with at least 10% annotation.

    """
    return (
        row.selection_source != "mask"
        or row.annotated_fraction >= MIN_MASK_ONLY_FRACTION
    )


def iter_extraction_rows(atlas_path: Path) -> Iterator[AtlasRow]:
    """Filter the lossless atlas through the locked extraction rule.

    Yields:
        Otsu patches and mask-only patches with at least 10% annotation.

    """
    for row in iter_atlas_rows(atlas_path):
        if _is_extraction_eligible(row):
            yield row


def generate_dataset(
    *,
    atlas_path: Path,
    wsi_dir: Path,
    output_dir: Path,
    part_number: int,
    part_count: int = DEFAULT_PART_COUNT,
    pyvips_module: ModuleType | None = None,
    expected_wsi_count: int = EXPECTED_WSI_COUNT,
) -> int:
    """Extract one deterministic WSI range into a standalone patch shard.

    The atlas, rather than progress files, defines the restart point. If this
    run stops, repeat the same part number: completed parts are untouched and
    Otsu is never rerun.

    Returns:
        Number of patches written.

    Raises:
        RuntimeError: If the written count differs from the atlas count.
        ValueError: If WSI shape or atlas coordinates violate the contract.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inspect_atlas(
        atlas_path,
        expected_wsi_count=expected_wsi_count,
    )
    plans = plan_dataset_parts(atlas_path, part_count=part_count)
    if part_number < 1 or part_number > part_count:
        message = f"part_number must be between 1 and {part_count}"
        raise ValueError(message)
    plan = plans[part_number - 1]
    selected_wsi_ids = frozenset(plan.wsi_ids)

    pyvips = _load_module("pyvips") if pyvips_module is None else pyvips_module
    _disable_vips_operation_cache(pyvips)

    bin_path = output_dir / f"{plan.stem}.bin"
    csv_path = output_dir / f"{plan.stem}.csv"
    provenance_path = output_dir / f"{plan.stem}.json"
    partial_bin = bin_path.with_suffix(f"{bin_path.suffix}.partial")
    partial_csv = csv_path.with_suffix(f"{csv_path.suffix}.partial")
    checksum = 0
    payload_sha256 = hashlib.sha256()
    total_written = 0

    with (
        partial_bin.open("wb", buffering=8 * 1024 * 1024) as binary_file,
        partial_csv.open("w", newline="", encoding="utf-8") as metadata_file,
    ):
        binary_file.write(b"\x00" * HEADER_SIZE)
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=PATCH_CSV_COLUMNS,
        )
        writer.writeheader()

        grouped_rows = groupby(
            iter_extraction_rows(atlas_path),
            key=lambda row: row.wsi_id,
        )
        slide_number = 0
        for wsi_id, rows in grouped_rows:
            if wsi_id not in selected_wsi_ids:
                continue
            slide_number += 1
            wsi_path = wsi_dir / f"{wsi_id}.png"
            # PNG pixels are decoded from top to bottom. ``sequential`` matches
            # our y,x atlas order and avoids the larger buffers needed to jump
            # around the image. ``fail=True`` prevents a damaged WSI from being
            # accepted with silently missing pixels.
            slide = pyvips.Image.new_from_file(
                str(wsi_path),
                access="sequential",
                fail=True,
            )
            if int(slide.bands) != CHANNELS:
                message = f"WSI {wsi_id} has {slide.bands} channels; expected 3"
                raise ValueError(message)
            region = pyvips.Region.new(slide)
            slide_patch_count = 0

            for y, y_rows_iterator in groupby(rows, key=lambda row: row.y):
                y_rows = tuple(y_rows_iterator)
                for row in y_rows:
                    if (
                        row.x + PATCH_SIZE > slide.width
                        or row.y + PATCH_SIZE > slide.height
                    ):
                        message = (
                            f"Out-of-bounds atlas row for WSI {wsi_id}: "
                            f"({row.x}, {row.y})"
                        )
                        raise ValueError(message)

                # PNG decoding already advances through complete scanlines. One
                # fetch from the first through last selected x therefore avoids
                # one Python/libvips call per patch without holding the WSI or
                # several y rows in RAM. The NumPy strip is a zero-copy view of
                # this blob and is released before the next y value.
                span_start = y_rows[0].x
                span_end = y_rows[-1].x + PATCH_SIZE
                span_width = span_end - span_start
                blob = region.fetch(span_start, y, span_width, PATCH_SIZE)
                strip = np.frombuffer(blob, dtype=np.uint8)
                expected_values = PATCH_SIZE * span_width * CHANNELS
                if strip.size != expected_values:
                    message = (
                        f"WSI {wsi_id} row y={y} returned {strip.size} values; "
                        f"expected {expected_values}"
                    )
                    raise ValueError(message)
                strip = strip.reshape(PATCH_SIZE, span_width, CHANNELS)

                for row in y_rows:
                    offset = row.x - span_start
                    image = strip[:, offset : offset + PATCH_SIZE, :]
                    payload = np.ascontiguousarray(image.transpose(2, 0, 1)).tobytes()
                    binary_file.write(payload)
                    checksum = zlib.crc32(payload, checksum)
                    payload_sha256.update(payload)
                    writer.writerow(_atlas_row_dict(row, idx=total_written))
                    total_written += 1
                    slide_patch_count += 1
                    del payload, image
                del strip, blob, y_rows

            print(
                f"Part {plan.number}/{plan.total_parts}, WSI "
                f"{slide_number}/{len(plan.wsi_ids)}: {wsi_id} "
                f"-> {slide_patch_count} patches",
                flush=True,
            )
            # Drop our references at the WSI boundary. This is sufficient;
            # forcing Python's garbage collector after every patch would cost
            # time and would not release anything that is still referenced.
            del region, slide

        if total_written != plan.patch_count:
            message = (
                f"Part {plan.number} wrote {total_written} patches but its "
                f"atlas range contains {plan.patch_count}"
            )
            raise RuntimeError(message)
        binary_file.seek(0)
        binary_file.write(
            struct.pack(
                HEADER_FORMAT,
                MAGIC,
                checksum & 0xFFFFFFFF,
                total_written,
                CHANNELS,
                PATCH_SIZE,
                PATCH_SIZE,
                VERSION,
                LAYOUT,
            ),
        )

    partial_bin.replace(bin_path)
    partial_csv.replace(csv_path)
    _write_provenance(
        provenance_path=provenance_path,
        atlas_path=atlas_path,
        wsi_dir=wsi_dir,
        wsi_count=len(plan.wsi_ids),
        patch_count=total_written,
        crc32=checksum & 0xFFFFFFFF,
        binary_sha256=_sha256_file(bin_path),
        payload_sha256=payload_sha256.hexdigest(),
        part_plan=plan,
        merged_parts=None,
    )
    print(f"Part saved to {bin_path} and {csv_path}", flush=True)
    return total_written


def merge_dataset_parts(
    *,
    atlas_path: Path,
    parts_dir: Path,
    output_dir: Path,
    part_count: int = DEFAULT_PART_COUNT,
    expected_wsi_count: int = EXPECTED_WSI_COUNT,
) -> int:
    """Stream validated part payloads into the final ordered test shard.

    Merge never loads a whole part into memory. It copies fixed-size byte chunks,
    recalculates the final CRC/SHA, and rewrites local part indices as one global
    ``idx`` sequence.

    Returns:
        Number of patches in the merged shard.

    Raises:
        ValueError: If a part is missing, corrupt, or differs from its atlas range.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _wsi_count, expected_patch_count = inspect_atlas(
        atlas_path,
        expected_wsi_count=expected_wsi_count,
    )
    plans = plan_dataset_parts(atlas_path, part_count=part_count)
    atlas_sha256 = _sha256_file(atlas_path)

    bin_path = output_dir / BIN_NAME
    csv_path = output_dir / CSV_NAME
    provenance_path = output_dir / PROVENANCE_NAME
    partial_bin = bin_path.with_suffix(f"{bin_path.suffix}.partial")
    partial_csv = csv_path.with_suffix(f"{csv_path.suffix}.partial")
    atlas_rows = iter_extraction_rows(atlas_path)
    merged_crc = 0
    merged_payload_sha256 = hashlib.sha256()
    total_written = 0

    with (
        partial_bin.open("wb", buffering=COPY_CHUNK_BYTES) as merged_binary,
        partial_csv.open("w", newline="", encoding="utf-8") as merged_metadata,
    ):
        merged_binary.write(b"\x00" * HEADER_SIZE)
        writer = csv.DictWriter(
            merged_metadata,
            fieldnames=PATCH_CSV_COLUMNS,
        )
        writer.writeheader()

        for plan in plans:
            part_bin = parts_dir / f"{plan.stem}.bin"
            part_csv = parts_dir / f"{plan.stem}.csv"
            part_json = parts_dir / f"{plan.stem}.json"
            part_crc, part_patch_count = _read_and_validate_header(part_bin)
            if part_patch_count != plan.patch_count:
                message = (
                    f"{part_bin.name} contains {part_patch_count} patches; "
                    f"atlas plan expects {plan.patch_count}"
                )
                raise ValueError(message)
            expected_binary_sha256 = _validate_part_provenance(
                provenance_path=part_json,
                plan=plan,
                atlas_sha256=atlas_sha256,
                patch_count=part_patch_count,
                crc32=part_crc,
            )

            local_count = 0
            with part_csv.open(newline="", encoding="utf-8") as metadata_file:
                for raw_row in csv.DictReader(metadata_file):
                    local_idx = int(_required(raw_row, "idx"))
                    if local_idx != local_count:
                        message = (
                            f"{part_csv.name} idx {local_idx} is not expected "
                            f"local index {local_count}"
                        )
                        raise ValueError(message)
                    observed = _parse_atlas_row(raw_row)
                    expected = next(atlas_rows, None)
                    if observed != expected:
                        message = (
                            f"{part_csv.name} row {local_idx} differs from atlas: "
                            f"{observed!r} != {expected!r}"
                        )
                        raise ValueError(message)
                    writer.writerow(_atlas_row_dict(observed, idx=total_written))
                    local_count += 1
                    total_written += 1
            if local_count != part_patch_count:
                message = (
                    f"{part_csv.name} has {local_count} rows but its binary "
                    f"header declares {part_patch_count}"
                )
                raise ValueError(message)

            observed_part_crc = 0
            observed_binary_sha256 = hashlib.sha256()
            with part_bin.open("rb") as part_binary:
                # The provenance hash covers the header as well as the payload.
                # This binds the correctly named metadata to the correctly named
                # binary and catches an accidental swap between equal-sized parts.
                observed_binary_sha256.update(part_binary.read(HEADER_SIZE))
                remaining = part_patch_count * CHANNELS * PATCH_SIZE**2
                while remaining:
                    chunk = part_binary.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        message = f"Unexpected end of payload in {part_bin}"
                        raise ValueError(message)
                    merged_binary.write(chunk)
                    observed_part_crc = zlib.crc32(chunk, observed_part_crc)
                    observed_binary_sha256.update(chunk)
                    merged_crc = zlib.crc32(chunk, merged_crc)
                    merged_payload_sha256.update(chunk)
                    remaining -= len(chunk)
            if observed_part_crc & 0xFFFFFFFF != part_crc:
                message = f"CRC32 mismatch in {part_bin}"
                raise ValueError(message)
            if observed_binary_sha256.hexdigest() != expected_binary_sha256:
                message = (
                    f"SHA-256 mismatch between {part_bin.name} and {part_json.name}"
                )
                raise ValueError(message)

            print(
                f"Merged part {plan.number}/{plan.total_parts}: "
                f"{part_patch_count} patches",
                flush=True,
            )

        if total_written != expected_patch_count or next(atlas_rows, None) is not None:
            message = (
                f"Merged {total_written} patches but atlas contains "
                f"{expected_patch_count}"
            )
            raise ValueError(message)
        merged_binary.seek(0)
        merged_binary.write(
            struct.pack(
                HEADER_FORMAT,
                MAGIC,
                merged_crc & 0xFFFFFFFF,
                total_written,
                CHANNELS,
                PATCH_SIZE,
                PATCH_SIZE,
                VERSION,
                LAYOUT,
            ),
        )

    partial_bin.replace(bin_path)
    partial_csv.replace(csv_path)
    _write_provenance(
        provenance_path=provenance_path,
        atlas_path=atlas_path,
        wsi_dir=None,
        wsi_count=expected_wsi_count,
        patch_count=total_written,
        crc32=merged_crc & 0xFFFFFFFF,
        binary_sha256=_sha256_file(bin_path),
        payload_sha256=merged_payload_sha256.hexdigest(),
        part_plan=None,
        merged_parts=[plan.stem for plan in plans],
    )
    print(f"Merged dataset saved to {bin_path} and {csv_path}", flush=True)
    return total_written


def _write_provenance(
    *,
    provenance_path: Path,
    atlas_path: Path,
    wsi_dir: Path | None,
    wsi_count: int,
    patch_count: int,
    crc32: int,
    binary_sha256: str,
    payload_sha256: str,
    part_plan: PartPlan | None,
    merged_parts: Sequence[str] | None,
) -> None:
    payload = {
        "schema_version": "spec0017.masked_holdout_test.v4",
        "atlas": str(atlas_path),
        "atlas_sha256": _sha256_file(atlas_path),
        "wsi_dir": None if wsi_dir is None else str(wsi_dir),
        "wsi_count": wsi_count,
        "patch_count": patch_count,
        "patch_size": PATCH_SIZE,
        "channels": CHANNELS,
        "layout": LAYOUT.decode("ascii"),
        "selection_rule": (
            "lossless atlas: all full-grid patches with any annotated mask "
            "pixel, plus HSV saturation Otsu projected foreground fraction > 0.6"
        ),
        "extraction_rule": (
            "passes Otsu or annotated_fraction >= 0.10 for mask-only candidates"
        ),
        "mask_only_min_annotated_fraction": MIN_MASK_ONLY_FRACTION,
        "mask_classes_rgb": {
            "red": "tumor",
            "green": "stroma",
            "blue": "necrosis",
            "black": "unannotated/unknown",
        },
        "mask_pixel_rule": (
            "nonblack pixels count once in annotated_fraction and contribute to "
            "every RGB channel tied for their maximum"
        ),
        "mask_label_rule": (
            "largest class fraction; exact largest-fraction tie is ambiguous"
        ),
        "mask_annotations_exhaustive": False,
        "ordered_by": ["wsi_id", "y", "x"],
        "binary_crc32": crc32,
        "binary_sha256": binary_sha256,
        "binary_payload_sha256": payload_sha256,
    }
    if part_plan is not None:
        payload["part"] = {
            "number": part_plan.number,
            "total_parts": part_plan.total_parts,
            "first_wsi_id": part_plan.wsi_ids[0],
            "last_wsi_id": part_plan.wsi_ids[-1],
            "wsi_ids": list(part_plan.wsi_ids),
        }
    if merged_parts is not None:
        payload["merged_parts"] = list(merged_parts)
    partial_path = provenance_path.with_suffix(f"{provenance_path.suffix}.partial")
    partial_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    partial_path.replace(provenance_path)


def _read_and_validate_header(bin_path: Path) -> tuple[int, int]:
    """Check one part's fixed binary contract.

    Returns:
        Header CRC32 and patch count.

    Raises:
        ValueError: If the header or file size is invalid.

    """
    with bin_path.open("rb") as binary_file:
        header_bytes = binary_file.read(HEADER_SIZE)
    if len(header_bytes) != HEADER_SIZE:
        message = f"Incomplete 64-byte header in {bin_path}"
        raise ValueError(message)
    magic, crc32, patch_count, channels, height, width, version, layout = struct.unpack(
        HEADER_FORMAT,
        header_bytes,
    )
    observed_contract = (magic, channels, height, width, version, layout)
    expected_contract = (MAGIC, CHANNELS, PATCH_SIZE, PATCH_SIZE, VERSION, LAYOUT)
    if observed_contract != expected_contract:
        message = f"Unsupported patch-shard header in {bin_path}: {observed_contract!r}"
        raise ValueError(message)
    expected_size = HEADER_SIZE + patch_count * CHANNELS * PATCH_SIZE**2
    if bin_path.stat().st_size != expected_size:
        message = (
            f"Size mismatch for {bin_path}: expected {expected_size} bytes, "
            f"found {bin_path.stat().st_size}"
        )
        raise ValueError(message)
    return int(crc32), int(patch_count)


def _validate_part_provenance(
    *,
    provenance_path: Path,
    plan: PartPlan,
    atlas_sha256: str,
    patch_count: int,
    crc32: int,
) -> str:
    """Bind a named part binary to its atlas range before merge.

    CRC detects damaged payload bytes, but it cannot tell whether two otherwise
    valid part binaries were placed under each other's filenames. The recorded
    full-file SHA and exact WSI list close that gap.

    Returns:
        Expected SHA-256 of the complete named part binary.

    Raises:
        ValueError: If provenance does not describe this atlas and part exactly.

    """
    payload = cast(
        "dict[str, object]",
        json.loads(provenance_path.read_text(encoding="utf-8")),
    )
    part = payload.get("part")
    expected_part = {
        "number": plan.number,
        "total_parts": plan.total_parts,
        "first_wsi_id": plan.wsi_ids[0],
        "last_wsi_id": plan.wsi_ids[-1],
        "wsi_ids": list(plan.wsi_ids),
    }
    expected_fields = {
        "atlas_sha256": atlas_sha256,
        "patch_count": patch_count,
        "binary_crc32": crc32,
    }
    observed_fields = {name: payload.get(name) for name in expected_fields}
    if part != expected_part or observed_fields != expected_fields:
        message = f"{provenance_path.name} does not match its atlas part"
        raise ValueError(message)
    binary_sha256 = payload.get("binary_sha256")
    if not isinstance(binary_sha256, str) or len(binary_sha256) != SHA256_HEX_LENGTH:
        message = f"{provenance_path.name} has no valid binary SHA-256"
        raise ValueError(message)
    return binary_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wsi_label_digest(labels_by_wsi: Mapping[int, int]) -> str:
    digest = hashlib.sha256()
    for wsi_id in sorted(labels_by_wsi):
        digest.update(f"{wsi_id},{labels_by_wsi[wsi_id]}\n".encode())
    return digest.hexdigest()


def _configure_vips_temp(output_dir: Path) -> None:
    """Keep any libvips temporary files on Kaggle's writable working volume."""
    # Do not put scratch files inside ``dataset/``: that directory is the clean
    # artifact tree we will download and later publish. On Kaggle, scratch stays
    # beside it under /kaggle/working; local calls retain their isolated output.
    temp_root = KAGGLE_WORKING_DIR if KAGGLE_WORKING_DIR.is_dir() else output_dir
    temp_dir = temp_root / "tmp_vips"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    # The historical generator used this as a last-resort disk safeguard for
    # unusually large decoded images. Sequential reads should not need it, but
    # retaining it is cheap and keeps any spill on the writable volume above.
    os.environ.setdefault("VIPS_DISC_THRESHOLD", "3gb")


def _ensure_kaggle_vips() -> None:
    """Install the two historical WSI dependencies on a real Kaggle worker.

    Kaggle already provides NumPy and OpenCV, but the proven train/validation
    notebook had to install the libvips shared library and its Python binding.
    We first try the import so a future Kaggle image that already contains both
    pays no setup cost. Installation is Kaggle-only: importing or testing this
    file locally never changes the developer environment.
    """
    try:
        importlib.import_module("pyvips")
    except (ImportError, OSError):
        if not KAGGLE_WORKING_DIR.is_dir():
            return
        sys.modules.pop("pyvips", None)
    else:
        return

    print("Installing Kaggle WSI reader dependencies...", flush=True)
    subprocess.run(  # noqa: S603 -- fixed command, only on a Kaggle worker
        [APT_GET, "update"],
        check=True,
    )
    subprocess.run(  # noqa: S603 -- fixed command, only on a Kaggle worker
        [APT_GET, "install", "-y", "--no-install-recommends", "libvips"],
        check=True,
    )
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            f"pyvips=={PYVIPS_VERSION}",
        ],
        check=True,
    )
    importlib.invalidate_caches()
    # Fail here with a direct dependency error instead of beginning the atlas
    # and discovering a broken shared-library install after output is opened.
    importlib.import_module("pyvips")


def _disable_vips_operation_cache(pyvips: ModuleType) -> None:
    """Disable reuse of completed libvips operations for this one-pass reader.

    We never request a patch twice. Retaining old operations therefore consumes
    memory without avoiding WSI decoding. Sequential access still keeps its own
    small read-behind buffer for nearby scanlines.
    """
    cache_set_max = getattr(pyvips, "cache_set_max", None)
    cache_set_max_mem = getattr(pyvips, "cache_set_max_mem", None)
    cache_set_max_files = getattr(pyvips, "cache_set_max_files", None)
    if callable(cache_set_max):
        cache_set_max(0)
    if callable(cache_set_max_mem):
        cache_set_max_mem(0)
    if callable(cache_set_max_files):
        cache_set_max_files(0)


def _load_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        message = (
            f"Missing runtime dependency {name!r}. On Kaggle, install libvips "
            "and pyvips before running this script."
        )
        raise RuntimeError(message) from error


def _required(row: dict[str, str | None], column: str) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        message = f"Missing required CSV value {column!r}"
        raise ValueError(message)
    return value.strip()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    message = f"Expected boolean CSV value, got {value!r}"
    raise ValueError(message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Preserve the three example commands as separate lines in ``--help``.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=("atlas", "dataset", "merge"),
        default="atlas",
        help="Build the atlas (default), one dataset part, or merge all parts.",
    )
    parser.add_argument(
        "--competition-root",
        type=Path,
        help=(
            "Directory containing train.csv, train_images, and train_thumbnails. "
            "When omitted, resolve the known Kaggle mount locations."
        ),
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help=(
            "Supplemental masks; filenames select held-out WSIs and RGB pixels "
            "supply incomplete tissue annotations. When omitted, resolve the "
            "known Kaggle dataset mount locations."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Generated-file destination (default: /kaggle/working/dataset).",
    )
    parser.add_argument(
        "--atlas-path",
        type=Path,
        help="Existing/resumed atlas path; defaults to OUTPUT_DIR/atlas filename.",
    )
    parser.add_argument(
        "--resume-atlas-checkpoint-dir",
        type=Path,
        help=(
            "Directory containing the checkpoint CSV/JSON from an interrupted "
            "atlas run. Completed WSIs are validated and copied into OUTPUT_DIR."
        ),
    )
    parser.add_argument(
        "--part",
        type=int,
        help="One-based part to extract; required for --stage dataset.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=DEFAULT_PART_COUNT,
        help="Number of deterministic WSI parts (default: 5).",
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        help="Directory holding completed parts for merge; defaults to OUTPUT_DIR.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected generation stage.

    Returns:
        Process exit status zero after successful generation.

    Raises:
        ValueError: If a dataset run omits its required part number.
        RuntimeError: If an internal stage/input-resolution invariant is broken.

    """
    args = _parse_args(argv)
    stage = cast("str", args.stage)
    requested_competition_root = cast("Path | None", args.competition_root)
    requested_mask_dir = cast("Path | None", args.mask_dir)
    output_dir = cast("Path", args.output_dir)
    supplied_atlas = cast("Path | None", args.atlas_path)
    requested_resume_atlas_checkpoint_dir = cast(
        "Path | None",
        args.resume_atlas_checkpoint_dir,
    )
    part_number = cast("int | None", args.part)
    part_count = cast("int", args.parts)
    supplied_parts_dir = cast("Path | None", args.parts_dir)
    atlas_path = output_dir / ATLAS_NAME if supplied_atlas is None else supplied_atlas
    parts_dir = output_dir if supplied_parts_dir is None else supplied_parts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_vips_temp(output_dir)

    # Resolve inputs before installing anything. A bad attachment should fail in
    # a few seconds with the paths we searched, not after apt/pip setup.
    competition_root: Path | None = None
    mask_dir: Path | None = None
    resume_atlas_checkpoint_dir: Path | None = None
    if stage in {"atlas", "dataset"}:
        competition_root = resolve_competition_root(
            requested_competition_root,
            require_atlas_inputs=stage == "atlas",
        )
        print(f"Resolved UBC-OCEAN root: {competition_root}")
    if stage == "atlas":
        mask_dir = resolve_mask_dir(requested_mask_dir)
        print(f"Resolved supplemental masks: {mask_dir}")
        resume_atlas_checkpoint_dir = resolve_atlas_checkpoint_dir(
            requested_resume_atlas_checkpoint_dir,
        )
        if resume_atlas_checkpoint_dir is not None:
            print(f"Resolved atlas checkpoint: {resume_atlas_checkpoint_dir}")

    if stage != "merge":
        _ensure_kaggle_vips()

    if stage == "atlas":
        if competition_root is None or mask_dir is None:
            message = "Atlas input resolution did not run"
            raise RuntimeError(message)
        slides = load_holdout_slides(competition_root / "train.csv", mask_dir)
        try:
            generate_atlas(
                slides=slides,
                thumbnail_dir=competition_root / "train_thumbnails",
                wsi_dir=competition_root / "train_images",
                mask_dir=mask_dir,
                atlas_path=atlas_path,
                resume_checkpoint_dir=resume_atlas_checkpoint_dir,
            )
        except Exception as error:
            if not _publish_atlas_checkpoint_after_error(
                atlas_path=atlas_path,
                slides=slides,
                error=error,
            ):
                raise
            traceback.print_exception(error)
            print(
                "Atlas is INCOMPLETE. Kaggle will publish the whole-WSI "
                "checkpoint instead of discarding completed work; attach this "
                "output to the next atlas run to resume.",
                flush=True,
            )
            return 0
        inspect_atlas(atlas_path)
        print_part_plan(plan_dataset_parts(atlas_path, part_count=part_count))
    elif stage == "dataset":
        if competition_root is None:
            message = "Dataset input resolution did not run"
            raise RuntimeError(message)
        if part_number is None:
            message = "--part is required for --stage dataset"
            raise ValueError(message)
        generate_dataset(
            atlas_path=atlas_path,
            wsi_dir=competition_root / "train_images",
            output_dir=output_dir,
            part_number=part_number,
            part_count=part_count,
        )
    else:
        merge_dataset_parts(
            atlas_path=atlas_path,
            parts_dir=parts_dir,
            output_dir=output_dir,
            part_count=part_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
