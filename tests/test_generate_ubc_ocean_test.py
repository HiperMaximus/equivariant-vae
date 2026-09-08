# Copyright 2026 HiperMaximus
"""Tests for the lean masked-holdout atlas and dataset generator."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess  # noqa: S404
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import kaggle.generate_ubc_ocean_test as generator
import numpy as np
import pytest
import torch
from kaggle.generate_ubc_ocean_test import (
    ATLAS_CHECKPOINT_NAME,
    ATLAS_CHECKPOINT_STATE_NAME,
    ATLAS_COLUMNS,
    ATLAS_INCOMPLETE_NAME,
    BIN_NAME,
    CHANNELS,
    CSV_NAME,
    PATCH_PIXELS,
    PROVENANCE_NAME,
    AtlasRow,
    HoldoutSlide,
    generate_atlas,
    generate_dataset,
    inspect_atlas,
    iter_atlas_rows,
    iter_tissue_coordinates,
    load_holdout_slides,
    mask_strip_fractions,
    merge_dataset_parts,
    plan_dataset_parts,
    resolve_atlas_checkpoint_dir,
)

from eqvae.data.patch_shards import PatchShard, PatchShardSpec

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from numpy.typing import NDArray

# These are deliberately small test-fixture values, not dataset settings.
# Two patches are enough to prove ordering without writing a large test file.
SYNTHETIC_PATCH_COUNT = 2
ADDITIVE_ATLAS_PATCH_COUNT = 3
FILTERED_EXTRACTION_PATCH_COUNT = 3
# Distinct channel values make a swapped or duplicated patch immediately visible.
LEFT_PATCH_RED = 17
RIGHT_PATCH_GREEN = 29
RIGHTMOST_PATCH_BLUE = 41
BELOW_MASK_ONLY_FRACTION = 0.05
MINIMUM_MASK_ONLY_FRACTION = 0.10
# The fake cv2 module uses arbitrary sentinel numbers. They only verify that the
# generator requests BGR->HSV and binary+Otsu; real OpenCV supplies its own values.
FAKE_CV2_BGR_TO_HSV = 2
FAKE_CV2_BINARY_OTSU = 12
MAX_UINT8_VALUE = 255


def test_kaggle_bootstrap_installs_only_missing_vips_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single uploaded script must repair Kaggle's known missing WSI stack.

    The historical notebook installed only libvips/pyvips. This test proves the
    new bootstrap preserves that narrow setup, pins the Python binding, and sets
    scratch space outside the clean ``dataset/`` artifact tree before import.
    """
    working_dir = tmp_path / "working"
    output_dir = working_dir / "dataset"
    working_dir.mkdir()
    commands: list[list[str]] = []

    def fake_import_module(name: str) -> ModuleType:
        if name != "pyvips":
            message = f"Unexpected import {name}"
            raise AssertionError(message)
        if not commands:
            raise ModuleNotFoundError(name)
        return ModuleType(name)

    def fake_run(
        command: list[str],
        *,
        check: bool,
    ) -> object:
        assert check is True
        commands.append(command)
        return object()

    for variable in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(variable, "test-original")
    monkeypatch.delenv("VIPS_DISC_THRESHOLD", raising=False)
    monkeypatch.setattr(generator, "KAGGLE_WORKING_DIR", working_dir)
    monkeypatch.setattr(generator.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(generator.subprocess, "run", fake_run)

    generator._configure_vips_temp(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        output_dir,
    )
    generator._ensure_kaggle_vips()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    expected_temp = working_dir / "tmp_vips"
    assert generator.os.environ["TMPDIR"] == str(expected_temp)
    assert generator.os.environ["TEMP"] == str(expected_temp)
    assert generator.os.environ["TMP"] == str(expected_temp)
    assert generator.os.environ["VIPS_DISC_THRESHOLD"] == "3gb"
    assert not (output_dir / "tmp_vips").exists()
    assert commands == [
        [generator.APT_GET, "update"],
        [
            generator.APT_GET,
            "install",
            "-y",
            "--no-install-recommends",
            "libvips",
        ],
        [
            generator.sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            f"pyvips=={generator.PYVIPS_VERSION}",
        ],
    ]


@pytest.mark.parametrize("kaggle_worker", [False, True])
def test_vips_bootstrap_never_installs_locally_or_when_import_already_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kaggle_worker: bool,
) -> None:
    """Only a real Kaggle worker with a missing import may modify its runtime."""
    working_dir = tmp_path / "working"
    if kaggle_worker:
        working_dir.mkdir()

    def fake_import_module(name: str) -> ModuleType:
        if kaggle_worker:
            return ModuleType(name)
        raise ModuleNotFoundError(name)

    def reject_run(*_args: object, **_kwargs: object) -> object:
        message = "Dependency installation must not run in this case"
        raise AssertionError(message)

    monkeypatch.setattr(generator, "KAGGLE_WORKING_DIR", working_dir)
    monkeypatch.setattr(generator.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(generator.subprocess, "run", reject_run)

    generator._ensure_kaggle_vips()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_merge_stage_never_bootstraps_image_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging existing byte shards must work without OpenCV, pyvips, or apt."""

    def reject_bootstrap() -> None:
        message = "Merge must not install image dependencies"
        raise AssertionError(message)

    def fake_merge(**_kwargs: object) -> int:
        return 0

    def fake_configure(_path: Path) -> None:
        pass

    monkeypatch.setattr(generator, "_ensure_kaggle_vips", reject_bootstrap)
    monkeypatch.setattr(generator, "_configure_vips_temp", fake_configure)
    monkeypatch.setattr(generator, "merge_dataset_parts", fake_merge)

    result = generator.main(
        [
            "--stage",
            "merge",
            "--output-dir",
            str(tmp_path / "output"),
            "--atlas-path",
            str(tmp_path / "atlas.csv"),
        ],
    )

    assert result == 0


def test_current_kaggle_mount_layout_is_resolved_without_reading_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current Kaggle mounts include source type, owner, and dataset version.

    The failed version-1 atlas run proved that the old direct mask path is no
    longer guaranteed even though the remote metadata says the source is
    attached. Resolution should therefore use directory/file identity and must
    not open any large WSI or mask pixels merely to locate the two inputs.
    """
    input_root = tmp_path / "input"
    competition_root = input_root / "competitions" / generator.COMPETITION_NAME
    (competition_root / "train_images").mkdir(parents=True)
    (competition_root / "train_thumbnails").mkdir()
    (competition_root / "train.csv").write_text("image_id,label,is_tma\n")

    mask_dir = (
        input_root
        / "datasets"
        / generator.MASK_DATASET_OWNER
        / generator.MASK_DATASET_NAME
        / "versions"
        / "7"
    )
    mask_dir.mkdir(parents=True)
    for mask_number in range(generator.EXPECTED_WSI_COUNT):
        (mask_dir / f"{mask_number}.png").touch()

    monkeypatch.setattr(generator, "KAGGLE_INPUT_ROOT", input_root)

    assert generator.resolve_competition_root(None) == competition_root
    assert generator.resolve_mask_dir(None) == mask_dir


def test_explicit_input_paths_fail_instead_of_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mistyped resume override must not silently select another attachment."""
    input_root = tmp_path / "input"
    automatic_mask_dir = input_root / generator.MASK_DATASET_NAME
    automatic_mask_dir.mkdir(parents=True)
    for mask_number in range(generator.EXPECTED_WSI_COUNT):
        (automatic_mask_dir / f"{mask_number}.png").touch()
    monkeypatch.setattr(generator, "KAGGLE_INPUT_ROOT", input_root)

    missing_override = tmp_path / "wrong-mask-path"
    with pytest.raises(FileNotFoundError, match=str(missing_override)):
        generator.resolve_mask_dir(missing_override)


def test_dataset_stage_resolves_wsi_only_root_without_atlas_inputs(
    tmp_path: Path,
) -> None:
    """Extraction must not reopen labels or thumbnails already sealed in the atlas."""
    competition_root = tmp_path / "competition"
    (competition_root / "train_images").mkdir(parents=True)

    assert (
        generator.resolve_competition_root(
            competition_root,
            require_atlas_inputs=False,
        )
        == competition_root
    )
    with pytest.raises(FileNotFoundError, match=r"train\.csv"):
        generator.resolve_competition_root(competition_root)


def test_one_attached_atlas_checkpoint_is_discovered_by_stable_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed script kernel should not depend on Kaggle's changing mount path."""
    input_root = tmp_path / "input"
    checkpoint_dir = input_root / "kernels" / "owner" / "old-run" / "dataset"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / ATLAS_CHECKPOINT_NAME).touch()
    (checkpoint_dir / ATLAS_CHECKPOINT_STATE_NAME).touch()
    monkeypatch.setattr(generator, "KAGGLE_INPUT_ROOT", input_root)

    assert resolve_atlas_checkpoint_dir(None) == checkpoint_dir

    second_dir = input_root / "another-output"
    second_dir.mkdir()
    (second_dir / ATLAS_CHECKPOINT_NAME).touch()
    (second_dir / ATLAS_CHECKPOINT_STATE_NAME).touch()
    with pytest.raises(ValueError, match="multiple attached atlas checkpoints"):
        resolve_atlas_checkpoint_dir(None)


def test_atlas_kernel_metadata_and_single_source_build_contract(  # noqa: PLR0914, PLR0915
    tmp_path: Path,
) -> None:
    """Fresh atlas runs attach two raw sources; resume permits one exact third.

    ``run.py`` is intentionally generated from the readable atlas script instead
    of becoming a second implementation that can drift before a remote launch.
    """
    repository = Path(__file__).parents[1]
    kernel_dir = repository / "kaggle/kernels/ubc_ocean_test_atlas"
    metadata = cast(
        "dict[str, object]",
        json.loads(
            (kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"),
        ),
    )
    workflow = (repository / "scripts/kaggle_kernel.sh").read_text(
        encoding="utf-8",
    )
    generator_source = (repository / "kaggle/generate_ubc_ocean_test.py").read_text(
        encoding="utf-8",
    )

    assert metadata == {
        "id": "maximusshtefan/eqvae-ubc-ocean-test-atlas",
        "title": "eqvae UBC-OCEAN test atlas",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "sohier/ubc-ovarian-cancer-competition-supplemental-masks",
        ],
        "competition_sources": ["UBC-OCEAN"],
        "kernel_sources": [],
        "model_sources": [],
    }
    assert "build_ubc_ocean_test_atlas_kernel" in workflow
    assert 'cp "$ubc_ocean_test_generator" "$kernel_dir/run.py"' in workflow
    assert "guard_ubc_ocean_test_atlas_push_ready" in workflow
    assert "KAGGLE_UBC_OCEAN_TEST_ATLAS_READY = True" in generator_source
    assert generator.DEFAULT_OUTPUT_DIR == generator.KAGGLE_WORKING_DIR / "dataset"

    # Exercise the real action dispatch in an isolated miniature repository.
    isolated = tmp_path / "repository"
    isolated_script = isolated / "scripts/kaggle_kernel.sh"
    isolated_generator = isolated / "kaggle/generate_ubc_ocean_test.py"
    isolated_kernel = isolated / "kaggle/kernels/ubc_ocean_test_atlas"
    isolated_script.parent.mkdir(parents=True)
    isolated_generator.parent.mkdir(parents=True)
    isolated_kernel.mkdir(parents=True)
    shutil.copy2(repository / "scripts/kaggle_kernel.sh", isolated_script)
    shutil.copy2(repository / "kaggle/generate_ubc_ocean_test.py", isolated_generator)
    shutil.copy2(kernel_dir / "kernel-metadata.json", isolated_kernel)
    bash = shutil.which("bash")
    assert bash is not None

    build = subprocess.run(  # noqa: S603
        (bash, str(isolated_script), "build", "kaggle/kernels/ubc_ocean_test_atlas"),
        cwd=isolated,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    isolated_run = isolated_kernel / "run.py"
    assert isolated_run.read_bytes() == isolated_generator.read_bytes()

    isolated_run.write_text("# stale\n", encoding="utf-8")
    stale = subprocess.run(  # noqa: S603
        (
            bash,
            str(isolated_script),
            "validate",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=isolated,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale.returncode != 0
    assert "does not match" in stale.stderr

    shutil.copy2(isolated_generator, isolated_run)
    resume_metadata = dict(metadata)
    resume_metadata["dataset_sources"] = [
        "sohier/ubc-ovarian-cancer-competition-supplemental-masks",
        "maximusshtefan/eqvae-ubc-ocean-test-atlas-checkpoint",
    ]
    (isolated_kernel / "kernel-metadata.json").write_text(
        f"{json.dumps(resume_metadata)}\n",
        encoding="utf-8",
    )
    fake_bin = isolated / "fake-bin"
    fake_bin.mkdir()
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'Kernel version 1 successfully pushed.  Please check "
        "progress at https://www.kaggle.com/code/professor-account/"
        "eqvae-ubc-ocean-test-atlas'\n",
        encoding="utf-8",
    )
    fake_kaggle.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["KAGGLE_PUSH_CONFIRMED"] = "1"
    environment["KAGGLE_FULL_DATASET_CONFIRMED"] = "1"
    environment["KAGGLE_DISABLE_FRESH_OAUTH"] = "1"
    environment["KAGGLE_USERNAME"] = "professor-account"
    environment["PYTHON"] = str(repository / ".venv/bin/python")
    environment["EQVAE_KAGGLE_LAUNCH_RECEIPT_ROOT"] = str(
        tmp_path / "launch-receipts",
    )
    resume_ready = subprocess.run(  # noqa: S603
        (
            bash,
            str(isolated_script),
            "push",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert resume_ready.returncode == 0, resume_ready.stderr

    wrong_resume_metadata = dict(resume_metadata)
    wrong_resume_metadata["dataset_sources"] = [
        "sohier/ubc-ovarian-cancer-competition-supplemental-masks",
        "someone/untrusted-checkpoint",
    ]
    (isolated_kernel / "kernel-metadata.json").write_text(
        f"{json.dumps(wrong_resume_metadata)}\n",
        encoding="utf-8",
    )
    wrong_resume = subprocess.run(  # noqa: S603
        (
            bash,
            str(isolated_script),
            "push",
            "kaggle/kernels/ubc_ocean_test_atlas",
        ),
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_resume.returncode != 0
    assert "exact private atlas-checkpoint dataset" in wrong_resume.stderr

    invalid_metadata = dict(metadata)
    invalid_metadata["enable_gpu"] = "true"
    (isolated_kernel / "kernel-metadata.json").write_text(
        f"{json.dumps(invalid_metadata)}\n",
        encoding="utf-8",
    )
    guarded = subprocess.run(  # noqa: S603
        (bash, str(isolated_script), "push", "kaggle/kernels/ubc_ocean_test_atlas"),
        cwd=isolated,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert guarded.returncode != 0
    assert "enable_gpu must be 'false'" in guarded.stderr


@dataclass
class _FakeSlide:
    pixels: NDArray[np.uint8]

    @property
    def width(self) -> int:
        return cast("int", self.pixels.shape[1])

    @property
    def height(self) -> int:
        return cast("int", self.pixels.shape[0])

    @property
    def bands(self) -> int:
        return cast("int", self.pixels.shape[2])


@dataclass
class _FakeRegion:
    slide: _FakeSlide
    fetch_calls: list[tuple[int, int, int, int]]

    def fetch(self, x: int, y: int, width: int, height: int) -> bytes:
        self.fetch_calls.append((x, y, width, height))
        return np.ascontiguousarray(
            self.slide.pixels[y : y + height, x : x + width],
        ).tobytes()


def _fake_pyvips(  # noqa: C901
    path: Path,
    pixels: NDArray[np.uint8],
    *,
    extra_images: Mapping[Path, NDArray[np.uint8]] | None = None,
    header_only_paths: Collection[Path] = (),
) -> ModuleType:
    module = ModuleType("fake_pyvips")
    images = {str(path): _FakeSlide(pixels)}
    header_paths = {str(header_path) for header_path in header_only_paths}
    open_calls: list[tuple[str, str | None, bool | None]] = []
    fetch_calls: list[tuple[int, int, int, int]] = []
    cache_limits: dict[str, int] = {}
    if extra_images is not None:
        images.update(
            {
                str(extra_path): _FakeSlide(extra)
                for extra_path, extra in extra_images.items()
            },
        )

    class Image:
        @staticmethod
        def new_from_file(
            requested_path: str,
            *,
            access: str | None = None,
            fail: bool | None = None,
        ) -> _FakeSlide:
            if requested_path not in images:
                message = "Generator must open an expected image"
                raise AssertionError(message)
            if any(cache_limits.get(name) != 0 for name in ("ops", "mem", "files")):
                message = "One-pass image reads must disable every libvips cache limit"
                raise AssertionError(message)
            expected_access = None if requested_path in header_paths else "sequential"
            if access != expected_access:
                message = f"Expected access={expected_access!r} for {requested_path}"
                raise AssertionError(message)
            if expected_access == "sequential" and fail is not True:
                message = "Sequential image access must fail on truncated PNG data"
                raise AssertionError(message)
            if expected_access is None and fail is not None:
                message = "Header-only WSI open must not request pixel-decoding options"
                raise AssertionError(message)
            open_calls.append((requested_path, access, fail))
            return images[requested_path]

    class Region:
        @staticmethod
        def new(requested_slide: _FakeSlide) -> _FakeRegion:
            return _FakeRegion(requested_slide, fetch_calls)

    def cache_set_max(value: int) -> None:
        cache_limits["ops"] = value

    def cache_set_max_mem(value: int) -> None:
        cache_limits["mem"] = value

    def cache_set_max_files(value: int) -> None:
        cache_limits["files"] = value

    module.__dict__["Image"] = Image
    module.__dict__["Region"] = Region
    module.__dict__["cache_set_max"] = cache_set_max
    module.__dict__["cache_set_max_mem"] = cache_set_max_mem
    module.__dict__["cache_set_max_files"] = cache_set_max_files
    module.__dict__["_open_calls"] = open_calls
    module.__dict__["_fetch_calls"] = fetch_calls
    module.__dict__["_cache_limits"] = cache_limits
    return module


def _atlas_row(  # noqa: PLR0913
    *,
    wsi_id: int,
    label: int,
    x: int,
    y: int,
    selection_source: str = "otsu",
    tumor_fraction: float = 0.0,
    stroma_fraction: float = 0.0,
    necrosis_fraction: float = 0.0,
    annotated_fraction: float | None = None,
) -> AtlasRow:
    fractions = (tumor_fraction, stroma_fraction, necrosis_fraction)
    resolved_annotated_fraction = (
        sum(fractions) if annotated_fraction is None else annotated_fraction
    )
    largest = max(fractions)
    winners = [
        name
        for name, fraction in zip(
            ("tumor", "stroma", "necrosis"),
            fractions,
            strict=True,
        )
        if fraction == largest and largest > 0.0
    ]
    mask_label = (
        "unknown" if not winners else winners[0] if len(winners) == 1 else "ambiguous"
    )
    return AtlasRow(
        wsi_id=wsi_id,
        label=label,
        x=x,
        y=y,
        selection_source=selection_source,
        mask_status=(
            "annotated" if resolved_annotated_fraction > 0.0 else "unannotated"
        ),
        mask_label=mask_label,
        annotated_fraction=resolved_annotated_fraction,
        tumor_fraction=tumor_fraction,
        stroma_fraction=stroma_fraction,
        necrosis_fraction=necrosis_fraction,
    )


def _atlas_row_values(row: AtlasRow) -> dict[str, object]:
    return {
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


def _write_raw_atlas(atlas_path: Path, rows: Sequence[dict[str, object]]) -> None:
    with atlas_path.open("w", newline="", encoding="utf-8") as atlas_file:
        writer = csv.DictWriter(atlas_file, fieldnames=ATLAS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)  # pyright: ignore[reportArgumentType]


def _write_atlas(atlas_path: Path, rows: Sequence[AtlasRow]) -> None:
    _write_raw_atlas(atlas_path, [_atlas_row_values(row) for row in rows])


def _fake_cv2(
    thumbnail_path: Path,
    tissue_mask: NDArray[np.uint8],
) -> ModuleType:
    module = ModuleType("fake_cv2")
    module.__dict__.update(
        {
            "IMREAD_COLOR": 1,
            "COLOR_BGR2HSV": FAKE_CV2_BGR_TO_HSV,
            "THRESH_BINARY": 4,
            "THRESH_OTSU": 8,
        },
    )

    def imread(requested_path: str, mode: int) -> NDArray[np.uint8]:
        if requested_path != str(thumbnail_path) or mode != 1:
            message = "Atlas must read the expected thumbnail in color"
            raise AssertionError(message)
        return np.zeros((*tissue_mask.shape, 3), dtype=np.uint8)

    def cvt_color(
        thumbnail: NDArray[np.uint8],
        conversion: int,
    ) -> NDArray[np.uint8]:
        if thumbnail.shape[2] != CHANNELS or conversion != FAKE_CV2_BGR_TO_HSV:
            message = "Atlas must convert the BGR thumbnail to HSV"
            raise AssertionError(message)
        hsv = thumbnail.copy()
        hsv[:, :, 1] = tissue_mask
        return hsv

    def threshold(
        saturation: NDArray[np.uint8],
        lower: int,
        upper: int,
        mode: int,
    ) -> tuple[float, NDArray[np.uint8]]:
        if (
            not np.array_equal(saturation, tissue_mask)
            or lower != 0
            or upper != MAX_UINT8_VALUE
            or mode != FAKE_CV2_BINARY_OTSU
        ):
            message = "Atlas must apply binary Otsu to HSV saturation"
            raise AssertionError(message)
        return 0.0, tissue_mask

    module.__dict__["imread"] = imread
    module.__dict__["cvtColor"] = cvt_color
    module.__dict__["threshold"] = threshold
    return module


def _generate_two_single_patch_parts(tmp_path: Path) -> tuple[Path, Path]:
    """Create two equal-sized parts so merge tests can mutate their artifacts.

    Returns:
        Atlas path and directory containing the two complete parts.

    """
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(
        atlas_path,
        [
            _atlas_row(
                wsi_id=10,
                label=0,
                x=0,
                y=0,
                selection_source="mask+otsu",
                tumor_fraction=1.0,
            ),
            _atlas_row(wsi_id=20, label=1, x=0, y=0),
        ],
    )
    wsi_dir = tmp_path / "train_images"
    parts_dir = tmp_path / "parts"
    wsi_dir.mkdir()
    parts_dir.mkdir()

    first_path = wsi_dir / "10.png"
    second_path = wsi_dir / "20.png"
    first_path.touch()
    second_path.touch()
    first_pixels = np.full((256, 256, 3), LEFT_PATCH_RED, dtype=np.uint8)
    second_pixels = np.full((256, 256, 3), RIGHT_PATCH_GREEN, dtype=np.uint8)

    for part_number, path, pixels in (
        (1, first_path, first_pixels),
        (2, second_path, second_pixels),
    ):
        generate_dataset(
            atlas_path=atlas_path,
            wsi_dir=wsi_dir,
            output_dir=parts_dir,
            part_number=part_number,
            part_count=2,
            pyvips_module=_fake_pyvips(path, pixels),
            expected_wsi_count=2,
        )
    return atlas_path, parts_dir


def test_otsu_grid_keeps_only_regions_above_the_historical_threshold() -> None:
    """The test atlas must preserve the train/validation strict >60% rule.

    This derived mask fixture catches changes to projected-area scaling, grid order,
    or an accidental switch from strict ``>`` to inclusive ``>=``.
    """
    tissue_mask = np.zeros((4, 4), dtype=np.uint8)
    tissue_mask[0, 0:2] = 255
    tissue_mask[1, 0] = 255
    tissue_mask[0:2, 2] = 255

    coordinates = list(
        iter_tissue_coordinates(
            tissue_mask,
            wsi_width=512,
            wsi_height=512,
        ),
    )

    assert coordinates == [(0, 0)]


def test_mask_rgb_order_and_antialiased_edges_keep_official_classes() -> None:
    """Red/green/blue mean tumor/stroma/necrosis even at blended boundaries.

    A BGR swap or discarded edge would create incorrect patch-level tissue targets.
    """
    pixels = np.zeros((256, 4 * 256, 3), dtype=np.uint8)
    pixels[:, 0:256] = (255, 0, 0)
    pixels[:, 256:512] = (0, 255, 0)
    pixels[:, 512:768] = (0, 0, 255)
    # Real supplemental masks contain blended edge colors. Red is still the
    # official class here because it is the unique strongest RGB channel.
    pixels[:128, 768:1024] = (200, 50, 10)

    annotated, tumor, stroma, necrosis = mask_strip_fractions(
        pixels.tobytes(),
        grid_width=4 * 256,
    )

    assert annotated.tolist() == [1.0, 1.0, 1.0, 0.5]
    assert tumor.tolist() == [1.0, 0.0, 0.0, 0.5]
    assert stroma.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert necrosis.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_mask_rgb_ties_count_each_maximum_but_annotation_once() -> None:
    """Tied maxima represent overlap without inflating annotation coverage.

    Nonmaximum antialias channels remain excluded, while a two- or three-way
    maximum tie contributes to every tied class under the locked mask contract.
    """
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[0, 0] = (25, 25, 0)
    pixels[0, 1] = (10, 10, 10)
    pixels[0, 2] = (200, 50, 10)

    annotated, tumor, stroma, necrosis = mask_strip_fractions(
        pixels.tobytes(),
        grid_width=256,
    )

    unit = 1.0 / generator.PATCH_PIXELS
    assert annotated.tolist() == pytest.approx([3 * unit])
    assert tumor.tolist() == pytest.approx([3 * unit])
    assert stroma.tolist() == pytest.approx([2 * unit])
    assert necrosis.tolist() == pytest.approx([unit])


def test_patch_mask_label_uses_largest_fraction_and_marks_exact_ties() -> None:
    """Keep mixed fractions while preventing arbitrary labels on exact ties.

    Choosing the first class on a tie would silently inject a false single-label
    target into the later tissue probe.
    """
    mask_label = generator._mask_label  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    assert mask_label(0.4, 0.6, 0.0) == "stroma"
    assert mask_label(0.5, 0.5, 0.0) == "ambiguous"
    assert mask_label(0.0, 0.0, 0.0) == "unknown"


def test_atlas_is_additive_and_preserves_grid_order(tmp_path: Path) -> None:
    """The atlas is the ordered union of mask annotations and HSV/Otsu tissue.

    One patch is mask-only, one is selected by both methods, and one is Otsu-only.
    This proves neither source acts as a gate on the other.
    """
    tissue_mask = np.zeros((4, 4), dtype=np.uint8)
    tissue_mask[0, 2:4] = 255
    tissue_mask[1, 2] = 255
    tissue_mask[2, 0:2] = 255
    tissue_mask[3, 0] = 255
    wsi_dir = tmp_path / "train_images"
    thumbnail_dir = tmp_path / "train_thumbnails"
    mask_dir = tmp_path / "masks"
    wsi_dir.mkdir()
    thumbnail_dir.mkdir()
    mask_dir.mkdir()
    wsi_path = wsi_dir / "10.png"
    thumbnail_path = thumbnail_dir / "10_thumbnail.png"
    mask_path = mask_dir / "10.png"
    wsi_path.touch()
    thumbnail_path.touch()
    mask_path.touch()
    atlas_path = tmp_path / "atlas.csv"
    mask_pixels = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_pixels[0:256, 0:256] = (255, 0, 0)
    mask_pixels[0:256, 256:512] = (0, 255, 0)
    fake_pyvips = _fake_pyvips(
        wsi_path,
        np.zeros((512, 512, 3), dtype=np.uint8),
        extra_images={mask_path: mask_pixels},
        header_only_paths=(wsi_path,),
    )

    patch_count = generate_atlas(
        slides=[HoldoutSlide(wsi_id=10, label=0)],
        thumbnail_dir=thumbnail_dir,
        wsi_dir=wsi_dir,
        mask_dir=mask_dir,
        atlas_path=atlas_path,
        cv2_module=_fake_cv2(thumbnail_path, tissue_mask),
        pyvips_module=fake_pyvips,
    )

    rows = list(iter_atlas_rows(atlas_path))
    assert patch_count == ADDITIVE_ATLAS_PATCH_COUNT
    assert [(row.x, row.y) for row in rows] == [
        (0, 0),
        (256, 0),
        (0, 256),
    ]
    assert [row.selection_source for row in rows] == [
        "mask",
        "mask+otsu",
        "otsu",
    ]
    assert [row.mask_label for row in rows] == [
        "tumor",
        "stroma",
        "unknown",
    ]
    assert fake_pyvips.__dict__["_cache_limits"] == {
        "ops": 0,
        "mem": 0,
        "files": 0,
    }
    assert fake_pyvips.__dict__["_open_calls"] == [
        (str(wsi_path), None, None),
        (str(mask_path), "sequential", True),
    ]


def test_atlas_resumes_after_last_complete_wsi_and_discards_partial_tail(  # noqa: PLR0914
    tmp_path: Path,
) -> None:
    """A failed next WSI must not force completed slides through Otsu again.

    The JSON state commits the safe CSV byte boundary only after a whole WSI.
    Extra bytes model a crash during the next slide; the resumed run truncates
    them, validates the retained prefix, and opens only the unfinished WSI.
    """
    slides = [HoldoutSlide(wsi_id=10, label=0), HoldoutSlide(wsi_id=20, label=1)]
    wsi_dir = tmp_path / "train_images"
    thumbnail_dir = tmp_path / "train_thumbnails"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "output"
    for directory in (wsi_dir, thumbnail_dir, mask_dir, output_dir):
        directory.mkdir()
    wsi_10 = wsi_dir / "10.png"
    wsi_20 = wsi_dir / "20.png"
    thumbnail_10 = thumbnail_dir / "10_thumbnail.png"
    thumbnail_20 = thumbnail_dir / "20_thumbnail.png"
    mask_10 = mask_dir / "10.png"
    mask_20 = mask_dir / "20.png"
    atlas_path = output_dir / generator.ATLAS_NAME
    tissue_mask = np.full((1, 1), MAX_UINT8_VALUE, dtype=np.uint8)
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)

    # WSI 10 completes, then WSI 20 fails its dimension check before Otsu.
    first_pyvips = _fake_pyvips(
        wsi_10,
        pixels,
        extra_images={
            wsi_20: pixels,
            mask_10: pixels,
            mask_20: np.zeros((128, 256, 3), dtype=np.uint8),
        },
        header_only_paths=(wsi_10, wsi_20),
    )
    with pytest.raises(ValueError, match="do not match WSI") as failure:
        generate_atlas(
            slides=slides,
            thumbnail_dir=thumbnail_dir,
            wsi_dir=wsi_dir,
            mask_dir=mask_dir,
            atlas_path=atlas_path,
            cv2_module=_fake_cv2(thumbnail_10, tissue_mask),
            pyvips_module=first_pyvips,
        )

    checkpoint_path = output_dir / ATLAS_CHECKPOINT_NAME
    state_path = output_dir / ATLAS_CHECKPOINT_STATE_NAME
    state = cast(
        "dict[str, object]",
        json.loads(state_path.read_text(encoding="utf-8")),
    )
    assert state["completed_wsi_ids"] == [10]
    assert state["patch_count"] == 1

    # An attached checkpoint must be validated before the launcher turns an
    # error into a publishable partial result. Otherwise one corrupt output
    # could be reported as successful and recycled forever.
    original_state_text = state_path.read_text(encoding="utf-8")
    wrong_digest_state = dict(state)
    wrong_digest_state["rows_sha256"] = "0" * generator.SHA256_HEX_LENGTH
    state_path.write_text(json.dumps(wrong_digest_state), encoding="utf-8")
    with pytest.raises(ValueError, match="committed SHA-256"):
        generator._resume_atlas_checkpoint(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            atlas_path=atlas_path,
            slides=slides,
            resume_checkpoint_dir=None,
        )

    corrupt_state = dict(state)
    corrupt_state["patch_count"] = 999
    state_path.write_text(json.dumps(corrupt_state), encoding="utf-8")
    assert not generator._publish_atlas_checkpoint_after_error(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        atlas_path=atlas_path,
        slides=slides,
        error=failure.value,
    )
    state_path.write_text(original_state_text, encoding="utf-8")

    assert generator._publish_atlas_checkpoint_after_error(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        atlas_path=atlas_path,
        slides=slides,
        error=failure.value,
    )
    assert (output_dir / ATLAS_INCOMPLETE_NAME).is_file()

    with checkpoint_path.open("ab") as checkpoint_file:
        checkpoint_file.write(b"uncommitted next-WSI tail\n")
    assert checkpoint_path.stat().st_size > cast("int", state["csv_bytes"])

    # The resumed backend deliberately lacks WSI 10. Any accidental reread of
    # that completed slide therefore fails the test immediately.
    second_pyvips = _fake_pyvips(
        wsi_20,
        pixels,
        extra_images={mask_20: pixels},
        header_only_paths=(wsi_20,),
    )
    patch_count = generate_atlas(
        slides=slides,
        thumbnail_dir=thumbnail_dir,
        wsi_dir=wsi_dir,
        mask_dir=mask_dir,
        atlas_path=atlas_path,
        cv2_module=_fake_cv2(thumbnail_20, tissue_mask),
        pyvips_module=second_pyvips,
    )

    assert patch_count == SYNTHETIC_PATCH_COUNT
    assert [(row.wsi_id, row.x, row.y) for row in iter_atlas_rows(atlas_path)] == [
        (10, 0, 0),
        (20, 0, 0),
    ]
    assert second_pyvips.__dict__["_open_calls"] == [
        (str(wsi_20), None, None),
        (str(mask_20), "sequential", True),
    ]
    assert not checkpoint_path.exists()
    assert not state_path.exists()
    assert not (output_dir / ATLAS_INCOMPLETE_NAME).exists()


def test_atlas_generation_refuses_a_stale_final_output(tmp_path: Path) -> None:
    """An interrupted rerun must never publish beside an older completed atlas.

    Refusing the pre-existing final file prevents Kaggle's output from looking
    complete when only the new run's checkpoint marker describes its state.
    """
    atlas_path = tmp_path / generator.ATLAS_NAME
    old_contents = "authoritative old atlas\n"
    atlas_path.write_text(old_contents, encoding="utf-8")
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_atlas(
            slides=[],
            thumbnail_dir=tmp_path,
            wsi_dir=tmp_path,
            mask_dir=tmp_path,
            atlas_path=atlas_path,
            cv2_module=ModuleType("unused_cv2"),
            pyvips_module=_fake_pyvips(tmp_path / "unused.png", pixels),
        )

    assert atlas_path.read_text(encoding="utf-8") == old_contents
    assert not (tmp_path / ATLAS_CHECKPOINT_NAME).exists()
    assert not (tmp_path / ATLAS_INCOMPLETE_NAME).exists()


def test_atlas_keeps_subthreshold_mask_candidate_but_inspection_requires_eligible_wsi(
    tmp_path: Path,
) -> None:
    """The resumable atlas is lossless even when its sole patch is not extractable.

    Keeping the painted coordinate permits later threshold audits, while failing
    inspection prevents a nominal test WSI from silently disappearing from binaries.
    """
    tissue_mask = np.zeros((1, 1), dtype=np.uint8)
    wsi_dir = tmp_path / "train_images"
    thumbnail_dir = tmp_path / "train_thumbnails"
    mask_dir = tmp_path / "masks"
    wsi_dir.mkdir()
    thumbnail_dir.mkdir()
    mask_dir.mkdir()
    wsi_path = wsi_dir / "10.png"
    thumbnail_path = thumbnail_dir / "10_thumbnail.png"
    mask_path = mask_dir / "10.png"
    wsi_path.touch()
    thumbnail_path.touch()
    mask_path.touch()
    mask_pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    mask_pixels[0, 0] = (255, 0, 0)
    fake_pyvips = _fake_pyvips(
        wsi_path,
        np.zeros((256, 256, 3), dtype=np.uint8),
        extra_images={mask_path: mask_pixels},
        header_only_paths=(wsi_path,),
    )
    atlas_path = tmp_path / "atlas.csv"

    patch_count = generate_atlas(
        slides=[HoldoutSlide(wsi_id=10, label=0)],
        thumbnail_dir=thumbnail_dir,
        wsi_dir=wsi_dir,
        mask_dir=mask_dir,
        atlas_path=atlas_path,
        cv2_module=_fake_cv2(thumbnail_path, tissue_mask),
        pyvips_module=fake_pyvips,
    )
    rows = list(iter_atlas_rows(atlas_path))

    assert patch_count == 1
    assert len(rows) == 1
    assert rows[0].selection_source == "mask"
    assert rows[0].annotated_fraction == pytest.approx(1 / PATCH_PIXELS)
    with pytest.raises(ValueError, match="no extraction-eligible patches"):
        inspect_atlas(atlas_path, expected_wsi_count=1)


def test_holdout_selection_uses_mask_ids_and_rejects_tma(tmp_path: Path) -> None:
    """Mask filenames define the cohort but TMA status remains a hard exclusion.

    A selected TMA would break the established 513-slide non-TMA partition, so the
    official metadata must veto it instead of silently producing a different test set.
    """
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "10.png").touch()
    (mask_dir / "20.png").touch()
    labels_csv = tmp_path / "train.csv"
    labels_csv.write_text(
        "image_id,label,is_tma\n10,CC,False\n20,MC,False\n30,HGSC,False\n",
        encoding="utf-8",
    )

    slides = load_holdout_slides(labels_csv, mask_dir, expected_wsi_count=2)

    assert [(slide.wsi_id, slide.label) for slide in slides] == [(10, 0), (20, 4)]

    labels_csv.write_text(
        "image_id,label,is_tma\n10,CC,False\n20,MC,True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marked as TMA"):
        load_holdout_slides(labels_csv, mask_dir, expected_wsi_count=2)


def test_ordered_atlas_writes_a_crc_valid_patch_shard(tmp_path: Path) -> None:
    """Atlas order must be the exact binary/CSV index order consumed by evaluation.

    The fake slide proves the writer preserves two spatially distinct patches and
    emits the existing header/CRC contract without requiring local libvips.
    """
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(
        atlas_path,
        [
            _atlas_row(
                wsi_id=10,
                label=2,
                x=0,
                y=0,
                selection_source="mask+otsu",
                tumor_fraction=0.25,
            ),
            _atlas_row(wsi_id=10, label=2, x=256, y=0),
        ],
    )

    pixels = np.zeros((256, 512, 3), dtype=np.uint8)
    pixels[:, :256, 0] = LEFT_PATCH_RED
    pixels[:, 256:, 1] = RIGHT_PATCH_GREEN
    wsi_dir = tmp_path / "train_images"
    wsi_dir.mkdir()
    fake_path = wsi_dir / "10.png"
    fake_path.touch()
    fake_pyvips = _fake_pyvips(fake_path, pixels)

    written = generate_dataset(
        atlas_path=atlas_path,
        wsi_dir=wsi_dir,
        output_dir=tmp_path,
        part_number=1,
        part_count=1,
        pyvips_module=fake_pyvips,
        expected_wsi_count=1,
    )
    part_stem = "ubc_ocean_test_part_01_of_01"
    shard = PatchShard(
        PatchShardSpec(
            bin_path=tmp_path / f"{part_stem}.bin",
            csv_path=tmp_path / f"{part_stem}.csv",
            validate_crc=True,
        ),
    )

    assert written == SYNTHETIC_PATCH_COUNT
    assert [record.x for record in shard.records] == [0, 256]
    assert torch.all(shard.read_uint8(0)[0] == LEFT_PATCH_RED)
    assert torch.all(shard.read_uint8(1)[1] == RIGHT_PATCH_GREEN)
    assert fake_pyvips.__dict__["_fetch_calls"] == [(0, 0, 512, 256)]
    with (tmp_path / f"{part_stem}.csv").open(
        newline="",
        encoding="utf-8",
    ) as metadata_file:
        metadata = list(csv.DictReader(metadata_file))
    assert metadata[0]["mask_label"] == "tumor"
    assert metadata[0]["tumor_fraction"] == "0.25"
    assert metadata[1]["mask_status"] == "unannotated"
    provenance = cast(
        "dict[str, object]",
        json.loads(
            (tmp_path / f"{part_stem}.json").read_text(
                encoding="utf-8",
            ),
        ),
    )
    with (tmp_path / f"{part_stem}.bin").open("rb") as binary_file:
        expected_sha256 = hashlib.file_digest(binary_file, "sha256").hexdigest()
    assert provenance["binary_sha256"] == expected_sha256


def test_extraction_drops_only_subthreshold_mask_only_candidates(
    tmp_path: Path,
) -> None:
    """The lossless atlas keeps mask specks while binaries use the 10% rule.

    Filtering before the shared row fetch prevents a nearly unpainted background
    patch from consuming binary space without discarding Otsu-qualified tissue.
    """
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(
        atlas_path,
        [
            _atlas_row(
                wsi_id=10,
                label=2,
                x=0,
                y=0,
                selection_source="mask",
                tumor_fraction=BELOW_MASK_ONLY_FRACTION,
            ),
            _atlas_row(
                wsi_id=10,
                label=2,
                x=256,
                y=0,
                selection_source="mask",
                tumor_fraction=MINIMUM_MASK_ONLY_FRACTION,
            ),
            _atlas_row(
                wsi_id=10,
                label=2,
                x=512,
                y=0,
                selection_source="mask",
                tumor_fraction=BELOW_MASK_ONLY_FRACTION,
            ),
            _atlas_row(wsi_id=10, label=2, x=768, y=0),
            _atlas_row(wsi_id=10, label=2, x=0, y=256),
        ],
    )
    pixels = np.zeros((512, 1024, 3), dtype=np.uint8)
    pixels[0:256, 0:256, 0] = LEFT_PATCH_RED
    pixels[0:256, 256:512, 1] = RIGHT_PATCH_GREEN
    pixels[0:256, 512:768, 0] = LEFT_PATCH_RED
    pixels[0:256, 768:1024, 2] = RIGHTMOST_PATCH_BLUE
    pixels[256:512, 0:256, 0] = LEFT_PATCH_RED
    wsi_dir = tmp_path / "train_images"
    wsi_dir.mkdir()
    fake_path = wsi_dir / "10.png"
    fake_path.touch()
    fake_pyvips = _fake_pyvips(fake_path, pixels)

    _wsi_count, eligible_count = inspect_atlas(
        atlas_path,
        expected_wsi_count=1,
    )
    written = generate_dataset(
        atlas_path=atlas_path,
        wsi_dir=wsi_dir,
        output_dir=tmp_path,
        part_number=1,
        part_count=1,
        pyvips_module=fake_pyvips,
        expected_wsi_count=1,
    )
    part_stem = "ubc_ocean_test_part_01_of_01"
    shard = PatchShard(
        PatchShardSpec(
            bin_path=tmp_path / f"{part_stem}.bin",
            csv_path=tmp_path / f"{part_stem}.csv",
            validate_crc=True,
        ),
    )

    assert eligible_count == FILTERED_EXTRACTION_PATCH_COUNT
    assert written == FILTERED_EXTRACTION_PATCH_COUNT
    assert [(record.x, record.y) for record in shard.records] == [
        (256, 0),
        (768, 0),
        (0, 256),
    ]
    assert torch.all(shard.read_uint8(0)[1] == RIGHT_PATCH_GREEN)
    assert torch.all(shard.read_uint8(1)[2] == RIGHTMOST_PATCH_BLUE)
    assert torch.all(shard.read_uint8(2)[0] == LEFT_PATCH_RED)
    assert fake_pyvips.__dict__["_fetch_calls"] == [
        (256, 0, 768, 256),
        (0, 256, 256, 256),
    ]

    merged_dir = tmp_path / "merged"
    merged_count = merge_dataset_parts(
        atlas_path=atlas_path,
        parts_dir=tmp_path,
        output_dir=merged_dir,
        part_count=1,
        expected_wsi_count=1,
    )
    merged = PatchShard(
        PatchShardSpec(
            bin_path=merged_dir / BIN_NAME,
            csv_path=merged_dir / CSV_NAME,
            validate_crc=True,
        ),
    )
    provenance = cast(
        "dict[str, object]",
        json.loads((merged_dir / PROVENANCE_NAME).read_text(encoding="utf-8")),
    )

    assert merged_count == FILTERED_EXTRACTION_PATCH_COUNT
    assert [(record.x, record.y) for record in merged.records] == [
        (256, 0),
        (768, 0),
        (0, 256),
    ]
    assert torch.all(merged.read_uint8(1)[2] == RIGHTMOST_PATCH_BLUE)
    assert provenance["schema_version"] == "spec0017.masked_holdout_test.v4"
    assert provenance["mask_label_rule"] == (
        "largest class fraction; exact largest-fraction tie is ambiguous"
    )
    assert provenance["mask_pixel_rule"] == (
        "nonblack pixels count once in annotated_fraction and contribute to "
        "every RGB channel tied for their maximum"
    )
    assert provenance["mask_only_min_annotated_fraction"] == pytest.approx(
        MINIMUM_MASK_ONLY_FRACTION,
    )
    assert provenance["extraction_rule"] == (
        "passes Otsu or annotated_fraction >= 0.10 for mask-only candidates"
    )


def test_default_five_part_plan_is_contiguous_complete_and_nonempty(
    tmp_path: Path,
) -> None:
    """Five uneven-work parts cover every WSI once without splitting one.

    Losing this boundary would reopen WSIs across runs and weaken simple part resume.
    """
    atlas_path = tmp_path / "atlas.csv"
    counts = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2]
    rows = [
        _atlas_row(
            wsi_id=index * 10,
            label=(index - 1) % 5,
            x=patch * 256,
            y=0,
        )
        for index, patch_count in enumerate(counts, start=1)
        for patch in range(patch_count)
    ]
    _write_atlas(atlas_path, rows)

    plans = plan_dataset_parts(atlas_path)

    assert [plan.number for plan in plans] == [1, 2, 3, 4, 5]
    assert [plan.wsi_ids for plan in plans] == [
        (10, 20),
        (30, 40),
        (50, 60),
        (70, 80),
        (90, 100),
    ]
    assert [plan.patch_count for plan in plans] == [5, 6, 7, 5, 6]
    assert [wsi_id for plan in plans for wsi_id in plan.wsi_ids] == list(
        range(10, 101, 10),
    )


def test_two_independent_parts_merge_back_to_exact_atlas_order(tmp_path: Path) -> None:
    """Separate Kaggle-style runs must reassemble into one ordinary shard.

    Order drift would make final CSV diagnoses and mask labels describe other pixels.
    """
    atlas_path, parts_dir = _generate_two_single_patch_parts(tmp_path)
    merged_dir = tmp_path / "merged"

    merged_count = merge_dataset_parts(
        atlas_path=atlas_path,
        parts_dir=parts_dir,
        output_dir=merged_dir,
        part_count=2,
        expected_wsi_count=2,
    )
    merged = PatchShard(
        PatchShardSpec(
            bin_path=merged_dir / BIN_NAME,
            csv_path=merged_dir / CSV_NAME,
            validate_crc=True,
        ),
    )

    assert merged_count == SYNTHETIC_PATCH_COUNT
    assert [record.wsi_id for record in merged.records] == ["10", "20"]
    assert torch.all(merged.read_uint8(0) == LEFT_PATCH_RED)
    assert torch.all(merged.read_uint8(1) == RIGHT_PATCH_GREEN)
    with (merged_dir / CSV_NAME).open(newline="", encoding="utf-8") as metadata_file:
        metadata = list(csv.DictReader(metadata_file))
    assert metadata[0]["selection_source"] == "mask+otsu"
    assert metadata[0]["mask_label"] == "tumor"
    assert metadata[1]["mask_label"] == "unknown"


def test_merge_requires_every_part_provenance_file(tmp_path: Path) -> None:
    """A part is incomplete until its identity/hash record is present.

    Accepting it without provenance could merge bytes from another atlas or WSI range.
    """
    atlas_path, parts_dir = _generate_two_single_patch_parts(tmp_path)
    (parts_dir / "ubc_ocean_test_part_02_of_02.json").unlink()

    with pytest.raises(FileNotFoundError):
        merge_dataset_parts(
            atlas_path=atlas_path,
            parts_dir=parts_dir,
            output_dir=tmp_path / "merged",
            part_count=2,
            expected_wsi_count=2,
        )


def test_merge_rejects_corrupt_part_payload(tmp_path: Path) -> None:
    """Changing one saved pixel must fail its streaming CRC before publish.

    Otherwise damaged Kaggle downloads could become the sealed evaluation artifact.
    """
    atlas_path, parts_dir = _generate_two_single_patch_parts(tmp_path)
    part_bin = parts_dir / "ubc_ocean_test_part_01_of_02.bin"
    with part_bin.open("r+b") as binary_file:
        binary_file.seek(64)
        original = binary_file.read(1)
        binary_file.seek(64)
        binary_file.write(bytes([original[0] ^ 0xFF]))

    with pytest.raises(ValueError, match="CRC32 mismatch"):
        merge_dataset_parts(
            atlas_path=atlas_path,
            parts_dir=parts_dir,
            output_dir=tmp_path / "merged",
            part_count=2,
            expected_wsi_count=2,
        )


def test_merge_rejects_equal_count_binary_swap(tmp_path: Path) -> None:
    """Per-part SHA binding prevents valid pixels receiving the wrong WSI IDs.

    Without the full-file binding, two equal-size parts could swap while retaining
    valid CRCs, silently attaching cancer and tissue labels to the wrong pixels.
    """
    atlas_path, parts_dir = _generate_two_single_patch_parts(tmp_path)
    first = parts_dir / "ubc_ocean_test_part_01_of_02.bin"
    second = parts_dir / "ubc_ocean_test_part_02_of_02.bin"
    temporary = parts_dir / "swap.tmp"
    first.replace(temporary)
    second.replace(first)
    temporary.replace(second)

    with pytest.raises(ValueError, match=r"does not match|SHA-256 mismatch"):
        merge_dataset_parts(
            atlas_path=atlas_path,
            parts_dir=parts_dir,
            output_dir=tmp_path / "merged",
            part_count=2,
            expected_wsi_count=2,
        )


@pytest.mark.parametrize(
    ("annotated", "changes", "message"),
    [
        (True, {"annotated_fraction": 0.25}, "cannot exceed"),
        (
            True,
            {"mask_status": "unannotated", "mask_label": "unknown"},
            "status/label disagree",
        ),
        (False, {"selection_source": "mask"}, "selection source"),
        (True, {"mask_label": "ambiguous"}, "status/label disagree"),
    ],
)
def test_atlas_reader_rejects_inconsistent_mask_metadata(
    tmp_path: Path,
    *,
    annotated: bool,
    changes: dict[str, object],
    message: str,
) -> None:
    """Downloaded atlas rows must keep source, status, label, and fractions coherent.

    Otherwise a resumable extraction could seal plausible pixels with corrupted
    supervised targets even though its coordinates and binary CRC remain valid.
    """
    row = (
        _atlas_row(
            wsi_id=10,
            label=0,
            x=0,
            y=0,
            selection_source="mask+otsu",
            tumor_fraction=0.5,
        )
        if annotated
        else _atlas_row(wsi_id=10, label=0, x=0, y=0)
    )
    raw_row = _atlas_row_values(row)
    raw_row.update(changes)
    atlas_path = tmp_path / "atlas.csv"
    _write_raw_atlas(atlas_path, [raw_row])

    with pytest.raises(ValueError, match=message):
        list(iter_atlas_rows(atlas_path))


def test_atlas_reader_accepts_overlapping_class_fractions(tmp_path: Path) -> None:
    """Tied pixels may raise the class sum above union annotation coverage.

    Rejecting this valid overlap would either lose real boundary patches or
    force their supervised tissue fractions into a false exclusive class.
    """
    row = _atlas_row(
        wsi_id=10,
        label=0,
        x=0,
        y=0,
        selection_source="mask",
        tumor_fraction=0.40,
        stroma_fraction=0.45,
        annotated_fraction=0.80,
    )
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(atlas_path, [row])

    assert list(iter_atlas_rows(atlas_path)) == [row]


def test_atlas_reader_rejects_nonsequential_coordinates(tmp_path: Path) -> None:
    """Strict atlas ordering protects sequential WSI access and binary identity.

    Reversing two coordinates would defeat the intended scan order and make a resumed
    extraction differ from the atlas contract, so the reader must fail before I/O.
    """
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(
        atlas_path,
        [
            _atlas_row(wsi_id=10, label=0, x=256, y=0),
            _atlas_row(wsi_id=10, label=0, x=0, y=0),
        ],
    )

    with pytest.raises(ValueError, match="ordered by wsi_id,y,x"):
        list(iter_atlas_rows(atlas_path))


def test_atlas_guards_reject_off_grid_and_inconsistent_labels(tmp_path: Path) -> None:
    """A resumed atlas cannot alter extraction geometry or one WSI's diagnosis.

    These guards keep a hand-edited/downloaded atlas from being sealed into a
    structurally valid shard whose coordinates or labels differ from generation.
    """
    atlas_path = tmp_path / "atlas.csv"
    _write_atlas(
        atlas_path,
        [_atlas_row(wsi_id=10, label=0, x=1, y=0)],
    )
    with pytest.raises(ValueError, match="align to the 256-pixel grid"):
        list(iter_atlas_rows(atlas_path))

    _write_atlas(
        atlas_path,
        [
            _atlas_row(wsi_id=10, label=0, x=0, y=0),
            _atlas_row(wsi_id=10, label=1, x=256, y=0),
        ],
    )
    with pytest.raises(ValueError, match="inconsistent labels"):
        inspect_atlas(atlas_path, expected_wsi_count=1)
