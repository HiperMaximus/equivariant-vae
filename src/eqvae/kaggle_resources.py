# Copyright 2026 HiperMaximus
"""Account-portable Kaggle resource and launch metadata helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_COMPONENT_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_UNVERSIONED_PARTS: Final = 2
_VERSIONED_PARTS: Final = 3
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_PUSH_CONFIRMATION_PATTERN: Final = re.compile(
    r"Kernel version (?P<version>[1-9][0-9]*) successfully pushed[.]\s+"
    r"Please check progress at (?P<url>https?://\S+)",
)
_REJECTED_PUSH_DETAIL_PATTERN: Final = re.compile(
    r"The following are not valid .+ and could not be added to the kernel:",
)
_SOURCE_FIELDS: Final = (
    "dataset_sources",
    "kernel_sources",
    "model_sources",
    "competition_sources",
)


def _validate_component(value: str, *, field: str) -> str:
    if not _COMPONENT_PATTERN.fullmatch(value):
        message = f"{field} must be a nonempty Kaggle identifier component"
        raise ValueError(message)
    return value


def _required[T](value: T | None, *, name: str) -> T:
    if value is None:
        message = f"internal CLI parser omitted required argument: {name}"
        raise ValueError(message)
    return value


@dataclass(frozen=True, slots=True)
class KaggleResourceRef:
    """Canonical owner/slug locator with an optional positive version."""

    owner: str
    slug: str
    version: int | None = None

    def __post_init__(self) -> None:
        """Validate each locator component.

        Raises:
            ValueError: If a component or version is invalid.

        """
        _validate_component(self.owner, field="owner")
        _validate_component(self.slug, field="slug")
        if self.version is not None and (
            isinstance(self.version, bool) or self.version < 1
        ):
            message = "version must be a positive integer"
            raise ValueError(message)

    @classmethod
    def parse(cls, value: str, *, allow_version: bool = True) -> KaggleResourceRef:
        """Parse ``owner/slug`` or ``owner/slug/version`` without inference.

        Returns:
            The exact structured resource reference.

        Raises:
            ValueError: If the reference grammar or version is invalid.

        """
        parts = value.split("/")
        if len(parts) == _UNVERSIONED_PARTS:
            return cls(owner=parts[0], slug=parts[1])
        if allow_version and len(parts) == _VERSIONED_PARTS:
            try:
                version = int(parts[2])
            except ValueError as error:
                message = "Kaggle resource version must be a positive integer"
                raise ValueError(message) from error
            return cls(owner=parts[0], slug=parts[1], version=version)
        message = "Kaggle resource reference must be owner/slug[/version]"
        raise ValueError(message)

    @property
    def canonical_id(self) -> str:
        """The unversioned canonical owner/slug identity."""
        return f"{self.owner}/{self.slug}"

    @property
    def reference(self) -> str:
        """The canonical identity, including the version when present."""
        if self.version is None:
            return self.canonical_id
        return f"{self.canonical_id}/{self.version}"

    def with_owner(self, owner: str) -> KaggleResourceRef:
        """Create the same slug/version under a different actor owner.

        Returns:
            A new resource reference with the requested owner.

        """
        return KaggleResourceRef(owner=owner, slug=self.slug, version=self.version)


def _read_object(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"expected a JSON object: {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sync_parent(path: Path) -> None:
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def source_locators(metadata: Mapping[str, object]) -> dict[str, list[str]]:
    """Validate source arrays and return exact, ordered copies.

    Kaggle model sources can have a deeper locator grammar, and competition
    sources commonly have no owner component. Therefore this function treats
    every nonempty string as an opaque canonical locator instead of trying to
    rewrite or normalize resource-owned identities.

    Returns:
        Exact ordered source strings keyed by Kaggle metadata field.

    Raises:
        TypeError: If a source field is not a list.
        ValueError: If a source entry is not an exact nonempty string.

    """
    result: dict[str, list[str]] = {}
    for field in _SOURCE_FIELDS:
        raw = metadata.get(field, [])
        if not isinstance(raw, list):
            message = f"{field} must be a list"
            raise TypeError(message)
        values: list[str] = []
        for value in cast("list[object]", raw):
            if not isinstance(value, str) or not value or value.strip() != value:
                message = f"{field} entries must be nonempty exact strings"
                raise ValueError(message)
            values.append(value)
        result[field] = values
    return result


def portable_kernel_metadata(
    metadata: Mapping[str, object],
    *,
    actor: str,
) -> dict[str, object]:
    """Change only a kernel's owner while preserving every input locator.

    Returns:
        A detached metadata object for the authenticated actor.

    Raises:
        TypeError: If the metadata id is not a string.
        AssertionError: If any source locator changes.

    """
    _validate_component(actor, field="authenticated Kaggle username")
    kernel_id = metadata.get("id")
    if not isinstance(kernel_id, str):
        message = "kernel metadata id must be owner/slug"
        raise TypeError(message)
    original = KaggleResourceRef.parse(kernel_id, allow_version=False)
    sources_before = source_locators(metadata)
    result = cast("dict[str, object]", json.loads(json.dumps(metadata)))
    result["id"] = original.with_owner(actor).canonical_id
    if source_locators(result) != sources_before:
        message = "portable snapshot changed a Kaggle source locator"
        raise AssertionError(message)
    return result


@dataclass(frozen=True, slots=True)
class PortableKernelSnapshot:
    """Local evidence describing an ephemeral account-portable snapshot."""

    source_metadata_sha256: str
    upload_metadata_sha256: str
    original_kernel_id: str
    upload_kernel_id: str
    source_locators: dict[str, list[str]]


def create_portable_kernel_snapshot(
    source_dir: Path,
    destination_dir: Path,
    *,
    actor: str,
) -> PortableKernelSnapshot:
    """Copy a kernel package and rewrite only the copied metadata owner.

    Returns:
        Hashes and identities describing the portable snapshot.

    Raises:
        FileNotFoundError: If the source package metadata is missing.
        FileExistsError: If the destination already exists.
        TypeError: If either kernel id is not a string after validation.

    """
    source_metadata_path = source_dir / "kernel-metadata.json"
    if not source_dir.is_dir() or not source_metadata_path.is_file():
        message = f"missing kernel package metadata: {source_metadata_path}"
        raise FileNotFoundError(message)
    if destination_dir.exists():
        message = f"portable snapshot destination already exists: {destination_dir}"
        raise FileExistsError(message)

    source_metadata = _read_object(source_metadata_path)
    portable_metadata = portable_kernel_metadata(source_metadata, actor=actor)
    shutil.copytree(source_dir, destination_dir)
    upload_metadata_path = destination_dir / "kernel-metadata.json"
    upload_metadata_path.write_text(
        json.dumps(portable_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    original_id = source_metadata.get("id")
    upload_id = portable_metadata.get("id")
    if not isinstance(original_id, str) or not isinstance(upload_id, str):
        message = "kernel ids must be strings"
        raise TypeError(message)
    return PortableKernelSnapshot(
        source_metadata_sha256=_sha256(source_metadata_path),
        upload_metadata_sha256=_sha256(upload_metadata_path),
        original_kernel_id=original_id,
        upload_kernel_id=upload_id,
        source_locators=source_locators(portable_metadata),
    )


def parse_kernel_push_confirmation(response: str) -> KaggleResourceRef:
    """Parse Kaggle's explicit accepted version and canonical URL.

    Returns:
        The exact canonical versioned kernel reference reported by Kaggle.

    Raises:
        ValueError: If the response has no unambiguous accepted kernel URL.

    """
    if "Kernel push error:" in response or _REJECTED_PUSH_DETAIL_PATTERN.search(
        response,
    ):
        message = "Kaggle push response reports rejected launch metadata"
        raise ValueError(message)
    matches = list(_PUSH_CONFIRMATION_PATTERN.finditer(response))
    if len(matches) != 1:
        message = "Kaggle did not return one explicit accepted kernel version and URL"
        raise ValueError(message)
    match = matches[0]
    parsed_url = urlparse(match.group("url"))
    if parsed_url.scheme != "https" or parsed_url.hostname not in {
        "kaggle.com",
        "www.kaggle.com",
    }:
        message = "Kaggle push confirmation returned an unexpected URL"
        raise ValueError(message)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if len(path_parts) != _VERSIONED_PARTS or path_parts[0] != "code":
        message = "Kaggle push confirmation URL must be /code/owner/slug"
        raise ValueError(message)
    return KaggleResourceRef(
        owner=path_parts[1],
        slug=path_parts[2],
        version=int(match.group("version")),
    )


def write_launch_receipt(
    *,
    source_dir: Path,
    upload_dir: Path,
    receipt_root: Path,
    accepted_reference: str,
) -> Path:
    """Write one immutable receipt for an accepted portable kernel launch.

    Returns:
        The new receipt path.

    Raises:
        TypeError: If either metadata id is not a string.
        ValueError: If identities, sources, or versions do not match.
        FileExistsError: If that exact launch receipt already exists.

    """
    source_metadata_path = source_dir / "kernel-metadata.json"
    upload_metadata_path = upload_dir / "kernel-metadata.json"
    source_metadata = _read_object(source_metadata_path)
    upload_metadata = _read_object(upload_metadata_path)
    original_id = source_metadata.get("id")
    upload_id = upload_metadata.get("id")
    if not isinstance(original_id, str) or not isinstance(upload_id, str):
        message = "kernel ids must be strings"
        raise TypeError(message)
    original_ref = KaggleResourceRef.parse(original_id, allow_version=False)
    upload_ref = KaggleResourceRef.parse(upload_id, allow_version=False)
    accepted_ref = KaggleResourceRef.parse(accepted_reference)
    if accepted_ref.version is None:
        message = "accepted kernel reference must include a positive version"
        raise ValueError(message)
    if upload_ref.owner != accepted_ref.owner:
        message = "accepted kernel owner does not match the authenticated actor"
        raise ValueError(message)
    if source_locators(source_metadata) != source_locators(upload_metadata):
        message = "portable upload changed source locators"
        raise ValueError(message)
    version_label = f"v{accepted_ref.version:04d}"
    destination = (
        receipt_root / accepted_ref.owner / accepted_ref.slug / f"{version_label}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "eqvae.kaggle_kernel_launch.v1",
        "actor": accepted_ref.owner,
        "original_kernel_id": original_ref.canonical_id,
        "requested_kernel_id": upload_ref.canonical_id,
        "kernel_id": accepted_ref.canonical_id,
        "accepted_version": accepted_ref.version,
        "kernel_reference": accepted_ref.reference,
        "source_locators": source_locators(upload_metadata),
        "source_metadata_sha256": _sha256(source_metadata_path),
        "upload_metadata_sha256": _sha256(upload_metadata_path),
        "source_files": _file_manifest(source_dir),
        "upload_files": _file_manifest(upload_dir),
    }
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        message = f"refusing to overwrite Kaggle launch receipt: {destination}"
        raise FileExistsError(message) from error
    _sync_parent(destination)
    return destination


def write_download_receipt(
    *,
    resource_kind: str,
    resource_reference: str,
    download_dir: Path,
    receipt_name: str,
) -> Path:
    """Bind downloaded files to their exact owner-qualified resource.

    Returns:
        The immutable receipt path inside the download directory.

    Raises:
        ValueError: If the kind, reference, or receipt name is invalid.
        FileNotFoundError: If the download directory has no regular files.
        FileExistsError: If the receipt already exists.

    """
    if resource_kind not in {"dataset", "kernel"}:
        message = "resource kind must be dataset or kernel"
        raise ValueError(message)
    reference = KaggleResourceRef.parse(resource_reference)
    if reference.version is None:
        message = "download provenance requires an exact resource version"
        raise ValueError(message)
    if Path(receipt_name).name != receipt_name or not receipt_name.endswith(".json"):
        message = "receipt name must be one local JSON filename"
        raise ValueError(message)
    destination = download_dir / receipt_name
    if destination.exists():
        message = f"refusing to overwrite download receipt: {destination}"
        raise FileExistsError(message)
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(download_dir.rglob("*")):
        if path.is_symlink():
            message = f"download receipt refuses symbolic links: {path}"
            raise ValueError(message)
        if path.is_file():
            files[path.relative_to(download_dir).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    if not files:
        message = f"download directory contains no files: {download_dir}"
        raise FileNotFoundError(message)
    receipt = {
        "schema_version": "eqvae.kaggle_download.v1",
        "resource_kind": resource_kind,
        "resource_owner": reference.owner,
        "resource_slug": reference.slug,
        "resource_version": reference.version,
        "resource_reference": reference.reference,
        "files": files,
    }
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


@dataclass(frozen=True, slots=True)
class _CliArgs:
    command: str
    source_dir: Path | None = None
    destination_dir: Path | None = None
    actor: str | None = None
    upload_dir: Path | None = None
    receipt_root: Path | None = None
    accepted_reference: str | None = None
    resource_kind: str | None = None
    resource_reference: str | None = None
    download_dir: Path | None = None
    receipt_name: str | None = None
    reference: str | None = None


def _parse_args(argv: Sequence[str] | None = None) -> _CliArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--source-dir", type=Path, required=True)
    snapshot.add_argument("--destination-dir", type=Path, required=True)
    snapshot.add_argument("--actor", required=True)
    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--source-dir", type=Path, required=True)
    receipt.add_argument("--upload-dir", type=Path, required=True)
    receipt.add_argument("--receipt-root", type=Path, required=True)
    receipt.add_argument("--accepted-reference", required=True)
    subparsers.add_parser("confirmation")
    download = subparsers.add_parser("download-receipt")
    download.add_argument(
        "--resource-kind",
        choices=("dataset", "kernel"),
        required=True,
    )
    download.add_argument("--resource-reference", required=True)
    download.add_argument("--download-dir", type=Path, required=True)
    download.add_argument("--receipt-name", required=True)
    validate_reference = subparsers.add_parser("validate-versioned-reference")
    validate_reference.add_argument("--reference", required=True)
    parsed = parser.parse_args(argv)
    return _CliArgs(
        command=cast("str", parsed.command),
        source_dir=cast("Path | None", getattr(parsed, "source_dir", None)),
        destination_dir=cast(
            "Path | None",
            getattr(parsed, "destination_dir", None),
        ),
        actor=cast("str | None", getattr(parsed, "actor", None)),
        upload_dir=cast("Path | None", getattr(parsed, "upload_dir", None)),
        receipt_root=cast("Path | None", getattr(parsed, "receipt_root", None)),
        accepted_reference=cast(
            "str | None",
            getattr(parsed, "accepted_reference", None),
        ),
        resource_kind=cast("str | None", getattr(parsed, "resource_kind", None)),
        resource_reference=cast(
            "str | None",
            getattr(parsed, "resource_reference", None),
        ),
        download_dir=cast("Path | None", getattr(parsed, "download_dir", None)),
        receipt_name=cast("str | None", getattr(parsed, "receipt_name", None)),
        reference=cast("str | None", getattr(parsed, "reference", None)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Create a portable snapshot or record its accepted canonical locator.

    Returns:
        Process exit status.

    Raises:
        ValueError: If a requested reference lacks an exact positive version.

    """
    args = _parse_args(argv)
    if args.command == "snapshot":
        snapshot = create_portable_kernel_snapshot(
            _required(args.source_dir, name="source-dir"),
            _required(args.destination_dir, name="destination-dir"),
            actor=_required(args.actor, name="actor"),
        )
        sys.stdout.write(f"{snapshot.upload_kernel_id}\n")
        return 0
    if args.command == "confirmation":
        confirmation = parse_kernel_push_confirmation(sys.stdin.read())
        sys.stdout.write(f"{confirmation.reference}\n")
        return 0
    if args.command == "download-receipt":
        destination = write_download_receipt(
            resource_kind=_required(args.resource_kind, name="resource-kind"),
            resource_reference=_required(
                args.resource_reference,
                name="resource-reference",
            ),
            download_dir=_required(args.download_dir, name="download-dir"),
            receipt_name=_required(args.receipt_name, name="receipt-name"),
        )
        sys.stdout.write(f"{destination}\n")
        return 0
    if args.command == "validate-versioned-reference":
        reference = KaggleResourceRef.parse(
            _required(args.reference, name="reference"),
        )
        if reference.version is None:
            message = "Kaggle resource reference requires a positive version"
            raise ValueError(message)
        sys.stdout.write(f"{reference.reference}\n")
        return 0
    if args.command != "receipt":
        message = f"unsupported Kaggle resource command: {args.command}"
        raise ValueError(message)
    destination = write_launch_receipt(
        source_dir=_required(args.source_dir, name="source-dir"),
        upload_dir=_required(args.upload_dir, name="upload-dir"),
        receipt_root=_required(args.receipt_root, name="receipt-root"),
        accepted_reference=_required(
            args.accepted_reference,
            name="accepted-reference",
        ),
    )
    sys.stdout.write(f"{destination}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
