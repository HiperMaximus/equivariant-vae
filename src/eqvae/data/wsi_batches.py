# Copyright 2026 HiperMaximus
"""Exact Spec 0021 coordinate-to-RGB batch streaming."""

from __future__ import annotations

import hashlib
import importlib
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from eqvae.data.latent_shards import LatentRowIdentity, WorkManifest

type WsiAccess = Literal["sequential", "random"]

PATCH_SIZE = 256
RGB_CHANNELS = 3
PATCH_BYTES = RGB_CHANNELS * PATCH_SIZE * PATCH_SIZE
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


class OpenedWSI(Protocol):
    """Minimal typed view of one decoded WSI."""

    @property
    def width(self) -> int:
        """Decoded pixel width."""
        ...

    @property
    def height(self) -> int:
        """Decoded pixel height."""
        ...

    @property
    def bands(self) -> int:
        """Decoded channel count."""
        ...

    def fetch(self, left: int, top: int, width: int, height: int) -> bytes:
        """Fetch one interleaved HWC byte rectangle."""
        ...


class WSIReader(Protocol):
    """Injectable seam used by real pyvips and array-backed fixture readers."""

    def open(self, path: Path, *, access: WsiAccess, fail: bool) -> OpenedWSI:
        """Open one WSI with an explicit access and decode-failure policy."""
        ...


class _VipsImage(Protocol):
    width: int
    height: int
    bands: int


class _VipsRegion(Protocol):
    def fetch(self, left: int, top: int, width: int, height: int) -> bytes: ...


class _VipsImageFactory(Protocol):
    def new_from_file(
        self,
        path: str,
        *,
        access: WsiAccess,
        fail: bool,
    ) -> _VipsImage: ...


class _VipsRegionFactory(Protocol):
    def new(self, image: _VipsImage) -> _VipsRegion: ...


class _PyVipsModule(Protocol):
    Image: _VipsImageFactory
    Region: _VipsRegionFactory

    def cache_set_max(self, value: int) -> None: ...

    def cache_set_max_mem(self, value: int) -> None: ...

    def cache_set_max_files(self, value: int) -> None: ...


@dataclass(frozen=True)
class WSIEvidence:
    """Durable byte identity for one fully streamed WSI selection."""

    wsi_id: int
    png_bytes: int
    png_sha256: str
    transcript_sha256: str


@dataclass(frozen=True)
class WSIPatchBatch:
    """One uint8 CHW batch that never crosses a WSI boundary."""

    row_start: int
    identities: tuple[LatentRowIdentity, ...]
    images_uint8: Tensor
    final_wsi_evidence: WSIEvidence | None


class _PyVipsOpenedWSI:
    def __init__(self, image: _VipsImage, region: _VipsRegion) -> None:
        self._image = image
        self._region = region

    @property
    def width(self) -> int:
        return int(self._image.width)

    @property
    def height(self) -> int:
        return int(self._image.height)

    @property
    def bands(self) -> int:
        return int(self._image.bands)

    def fetch(self, left: int, top: int, width: int, height: int) -> bytes:
        return self._region.fetch(left, top, width, height)


class PyVipsWSIReader:
    """Lazy pyvips adapter; importing this module never imports pyvips."""

    def __init__(self, module: _PyVipsModule | None = None) -> None:
        """Use an injected typed module or import pyvips lazily.

        Raises:
            RuntimeError: If no injected module and pyvips is unavailable.

        """
        if module is None:
            try:
                imported = importlib.import_module("pyvips")
            except ModuleNotFoundError as error:
                message = "pyvips is required for real WSI inference"
                raise RuntimeError(message) from error
            module = cast("_PyVipsModule", cast("object", imported))
        self._module = module
        module.cache_set_max(0)
        module.cache_set_max_mem(0)
        module.cache_set_max_files(0)

    def open(self, path: Path, *, access: WsiAccess, fail: bool) -> OpenedWSI:
        """Open one pyvips image and its region fetch adapter.

        Returns:
            The minimal typed opened-WSI view.

        """
        image = self._module.Image.new_from_file(
            str(path),
            access=access,
            fail=fail,
        )
        return _PyVipsOpenedWSI(image, self._module.Region.new(image))


class _ArrayOpenedWSI:
    def __init__(self, image: NDArray[np.uint8]) -> None:
        self._image = image

    @property
    def width(self) -> int:
        shape = cast("tuple[int, ...]", self._image.shape)
        return shape[1]

    @property
    def height(self) -> int:
        shape = cast("tuple[int, ...]", self._image.shape)
        return shape[0]

    @property
    def bands(self) -> int:
        shape = cast("tuple[int, ...]", self._image.shape)
        return shape[2]

    def fetch(self, left: int, top: int, width: int, height: int) -> bytes:
        crop = self._image[top : top + height, left : left + width, :]
        return np.ascontiguousarray(crop).tobytes(order="C")


class ArrayWSIReader:
    """Array-backed local-fixture reader keyed by integer PNG stem."""

    def __init__(self, images: Mapping[int, NDArray[np.uint8]]) -> None:
        """Store fixture HWC arrays without copying them."""
        self._images = dict(images)

    def open(self, path: Path, *, access: WsiAccess, fail: bool) -> OpenedWSI:
        """Resolve the integer PNG stem in the fixture mapping.

        Returns:
            An array-backed opened-WSI view.

        Raises:
            FileNotFoundError: If the PNG stem has no fixture array.
            ValueError: If the fixture is not an HWC uint8 array.

        """
        del access, fail
        try:
            wsi_id = int(path.stem)
            image = self._images[wsi_id]
        except (KeyError, ValueError) as error:
            message = f"No fixture WSI for {path.name}"
            raise FileNotFoundError(message) from error
        if image.ndim != RGB_CHANNELS or image.dtype != np.uint8:
            message = "Fixture WSI must be an HWC uint8 array"
            raise ValueError(message)
        return _ArrayOpenedWSI(image)


def iter_wsi_patch_batches(  # noqa: PLR0913
    *,
    manifest: WorkManifest,
    wsi_dir: Path,
    batch_size: int,
    start_row: int = 0,
    stop_row: int | None = None,
    reader: WSIReader | None = None,
) -> Iterator[WSIPatchBatch]:
    """Stream exact manifest patches in WSI-bounded uint8 CHW batches.

    Yields:
        Contiguous uint8 CHW tensors with their adjacent row identities.

    Raises:
        ValueError: If bounds, channels, byte sizes, or WSI boundaries differ.

    """
    if batch_size < 1:
        message = "batch_size must be positive"
        raise ValueError(message)
    resolved_stop = len(manifest.rows) if stop_row is None else stop_row
    selected_ranges = _selected_wsi_ranges(
        manifest,
        start_row=start_row,
        stop_row=resolved_stop,
    )
    active_reader = PyVipsWSIReader() if reader is None else reader
    for wsi_id, wsi_start, wsi_end in selected_ranges:
        path = wsi_dir / f"{wsi_id}.png"
        png_bytes, png_sha256 = _file_identity(path)
        opened = active_reader.open(path, access="sequential", fail=True)
        if opened.bands != RGB_CHANNELS:
            message = f"WSI {wsi_id} has {opened.bands} bands; expected 3"
            raise ValueError(message)
        yield from _iter_one_wsi(
            opened=opened,
            manifest=manifest,
            wsi_id=wsi_id,
            wsi_start=wsi_start,
            wsi_end=wsi_end,
            batch_size=batch_size,
            png_bytes=png_bytes,
            png_sha256=png_sha256,
        )


def iter_wsi_patch_interval(  # noqa: PLR0913
    *,
    manifest: WorkManifest,
    wsi_dir: Path,
    batch_size: int,
    start_row: int,
    stop_row: int,
    reader: WSIReader | None = None,
) -> Iterator[WSIPatchBatch]:
    """Stream one small partial-WSI interval without hashing the source PNG.

    This helper is reserved for bounded smoke checks. Production uses
    :func:`iter_wsi_patch_batches`, which requires complete WSI boundaries and
    emits durable WSI evidence.

    Yields:
        Contiguous uint8 CHW tensors for exactly ``[start_row, stop_row)``.

    Raises:
        ValueError: If the interval is empty, invalid, or crosses a WSI.

    """
    if batch_size < 1:
        message = "batch_size must be positive"
        raise ValueError(message)
    if not 0 <= start_row < stop_row <= len(manifest.rows):
        message = "Smoke manifest row interval is invalid"
        raise ValueError(message)
    wsi_ids = {row.identity.wsi_id for row in manifest.rows[start_row:stop_row]}
    if len(wsi_ids) != 1:
        message = "Smoke manifest row interval must stay within one WSI"
        raise ValueError(message)
    wsi_id = wsi_ids.pop()
    active_reader = PyVipsWSIReader() if reader is None else reader
    opened = active_reader.open(
        wsi_dir / f"{wsi_id}.png",
        access="sequential",
        fail=True,
    )
    if opened.bands != RGB_CHANNELS:
        message = f"WSI {wsi_id} has {opened.bands} bands; expected 3"
        raise ValueError(message)
    yield from _iter_one_wsi(
        opened=opened,
        manifest=manifest,
        wsi_id=wsi_id,
        wsi_start=start_row,
        wsi_end=stop_row,
        batch_size=batch_size,
        png_bytes=None,
        png_sha256=None,
    )


def read_wsi_patch(
    *,
    identity: LatentRowIdentity,
    wsi_dir: Path,
    reader: WSIReader | None = None,
) -> Tensor:
    """Independently random-open and read one exact sentinel crop.

    Returns:
        A copied contiguous uint8 CHW tensor.

    """
    active_reader = PyVipsWSIReader() if reader is None else reader
    opened = active_reader.open(
        wsi_dir / f"{identity.wsi_id}.png",
        access="random",
        fail=True,
    )
    _validate_bounds(opened, identity)
    blob = opened.fetch(identity.x, identity.y, PATCH_SIZE, PATCH_SIZE)
    return _chw_tensor(blob, span_width=PATCH_SIZE, offset=0).clone()


def _iter_one_wsi(  # noqa: PLR0913
    *,
    opened: OpenedWSI,
    manifest: WorkManifest,
    wsi_id: int,
    wsi_start: int,
    wsi_end: int,
    batch_size: int,
    png_bytes: int | None,
    png_sha256: str | None,
) -> Iterator[WSIPatchBatch]:
    transcript = hashlib.sha256()
    pending_images: list[Tensor] = []
    pending_identities: list[LatentRowIdentity] = []
    batch_start = wsi_start
    cursor = wsi_start
    while cursor < wsi_end:
        y = manifest.rows[cursor].identity.y
        y_end = cursor + 1
        while y_end < wsi_end and manifest.rows[y_end].identity.y == y:
            y_end += 1
        identities = tuple(row.identity for row in manifest.rows[cursor:y_end])
        for identity in identities:
            _validate_bounds(opened, identity)
        span_start = identities[0].x
        span_width = identities[-1].x + PATCH_SIZE - span_start
        blob = opened.fetch(span_start, y, span_width, PATCH_SIZE)
        for identity in identities:
            image = _chw_tensor(
                blob,
                span_width=span_width,
                offset=identity.x - span_start,
            ).clone()
            payload = image.numpy().tobytes(order="C")
            transcript.update(
                struct.pack(
                    "<QQQQ",
                    identity.atlas_row_index,
                    identity.wsi_id,
                    identity.x,
                    identity.y,
                ),
            )
            transcript.update(payload)
            pending_images.append(image)
            pending_identities.append(identity)
            processed = identity == manifest.rows[wsi_end - 1].identity
            if len(pending_images) == batch_size or processed:
                evidence = None
                if processed and png_bytes is not None and png_sha256 is not None:
                    evidence = WSIEvidence(
                        wsi_id=wsi_id,
                        png_bytes=png_bytes,
                        png_sha256=png_sha256,
                        transcript_sha256=transcript.hexdigest(),
                    )
                yield WSIPatchBatch(
                    row_start=batch_start,
                    identities=tuple(pending_identities),
                    images_uint8=torch.stack(pending_images),
                    final_wsi_evidence=evidence,
                )
                batch_start += len(pending_images)
                pending_images.clear()
                pending_identities.clear()
        cursor = y_end


def _selected_wsi_ranges(
    manifest: WorkManifest,
    *,
    start_row: int,
    stop_row: int,
) -> tuple[tuple[int, int, int], ...]:
    if not 0 <= start_row <= stop_row <= len(manifest.rows):
        message = "Requested manifest row interval is invalid"
        raise ValueError(message)
    boundaries = {0, len(manifest.rows)}
    for _wsi_id, start, end in manifest.wsi_ranges:
        boundaries.update((start, end))
    if start_row not in boundaries or stop_row not in boundaries:
        message = "WSI stream start/stop must be complete WSI boundaries"
        raise ValueError(message)
    return tuple(
        item
        for item in manifest.wsi_ranges
        if item[1] >= start_row and item[2] <= stop_row
    )


def _validate_bounds(opened: OpenedWSI, identity: LatentRowIdentity) -> None:
    if (
        identity.x + PATCH_SIZE > opened.width
        or identity.y + PATCH_SIZE > opened.height
    ):
        message = (
            f"Out-of-bounds coordinate for WSI {identity.wsi_id}: "
            f"({identity.x}, {identity.y})"
        )
        raise ValueError(message)


def _chw_tensor(blob: bytes, *, span_width: int, offset: int) -> Tensor:
    expected = PATCH_SIZE * span_width * RGB_CHANNELS
    if len(blob) != expected:
        message = f"WSI strip returned {len(blob)} values; expected {expected}"
        raise ValueError(message)
    strip = torch.frombuffer(bytearray(blob), dtype=torch.uint8).reshape(
        PATCH_SIZE,
        span_width,
        RGB_CHANNELS,
    )
    crop = strip[:, offset : offset + PATCH_SIZE, :]
    return crop.permute(2, 0, 1).contiguous()


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


__all__ = [
    "PATCH_BYTES",
    "PATCH_SIZE",
    "ArrayWSIReader",
    "OpenedWSI",
    "PyVipsWSIReader",
    "WSIEvidence",
    "WSIPatchBatch",
    "WSIReader",
    "iter_wsi_patch_batches",
    "iter_wsi_patch_interval",
    "read_wsi_patch",
]
