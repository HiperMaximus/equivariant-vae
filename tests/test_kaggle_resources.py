# Copyright 2026 HiperMaximus
"""Tests for account-portable Kaggle resource handling."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING, Final, cast

import pytest

from eqvae.kaggle_resources import (
    KaggleResourceRef,
    create_portable_kernel_snapshot,
    parse_kernel_push_confirmation,
    portable_kernel_metadata,
    write_download_receipt,
    write_launch_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path

_ACCEPTED_VERSION: Final = 4
_ACCEPTED_REFERENCE: Final = "professor/normalized-by-kaggle/4"


def _metadata() -> dict[str, object]:
    return {
        "id": "alice/pathology-job",
        "title": "Pathology job",
        "code_file": "run.py",
        "language": "python",
        "dataset_sources": ["alice/patches", "carol/private-checkpoint"],
        "kernel_sources": ["dave/upstream-kernel/7"],
        "model_sources": ["erin/model/pyTorch/default/3"],
        "competition_sources": ["UBC-OCEAN"],
    }


def _write_kernel(root: Path) -> Path:
    kernel = root / "kernel"
    kernel.mkdir()
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    (kernel / "run.py").write_text("print('ok')\n", encoding="utf-8")
    return kernel


def test_resource_ref_preserves_owner_slug_and_optional_version() -> None:
    """Canonical references retain their explicit owner and version."""
    assert KaggleResourceRef.parse("alice/data").reference == "alice/data"
    versioned = KaggleResourceRef.parse("alice/data/12")
    assert versioned.canonical_id == "alice/data"
    assert versioned.reference == "alice/data/12"
    assert versioned.with_owner("professor").reference == "professor/data/12"


@pytest.mark.parametrize(
    "value",
    ["data", "alice/data/not-a-version", "alice/data/0", "/data", "alice/"],
)
def test_resource_ref_rejects_ambiguous_or_invalid_locators(value: str) -> None:
    """Malformed or ambiguous references fail closed."""
    with pytest.raises(ValueError, match=r"Kaggle resource|version|component"):
        KaggleResourceRef.parse(value)


def test_portable_metadata_changes_only_kernel_actor() -> None:
    """Actor replacement cannot rewrite resource-owned source locators."""
    source = _metadata()
    portable = portable_kernel_metadata(source, actor="professor")

    assert portable["id"] == "professor/pathology-job"
    assert source["id"] == "alice/pathology-job"
    for field in (
        "dataset_sources",
        "kernel_sources",
        "model_sources",
        "competition_sources",
    ):
        assert portable[field] == source[field]


def test_snapshot_does_not_mutate_source_and_preserves_mixed_owner_inputs(
    tmp_path: Path,
) -> None:
    """Snapshotting leaves the source tree and mixed-owner inputs unchanged."""
    source = _write_kernel(tmp_path)
    destination = tmp_path / "upload"
    original_tree = {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }

    snapshot = create_portable_kernel_snapshot(
        source,
        destination,
        actor="professor",
    )

    after_tree = {
        path.relative_to(source): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after_tree == original_tree
    assert snapshot.original_kernel_id == "alice/pathology-job"
    assert snapshot.upload_kernel_id == "professor/pathology-job"
    assert snapshot.source_locators["dataset_sources"] == [
        "alice/patches",
        "carol/private-checkpoint",
    ]
    assert (destination / "run.py").read_bytes() == (source / "run.py").read_bytes()


def test_launch_receipt_saves_canonical_versioned_reference(tmp_path: Path) -> None:
    """Launch receipts retain the exact actor, sources, and accepted version."""
    source = _write_kernel(tmp_path)
    upload = tmp_path / "upload"
    create_portable_kernel_snapshot(source, upload, actor="professor")

    receipt_path = write_launch_receipt(
        source_dir=source,
        upload_dir=upload,
        receipt_root=tmp_path / "receipts",
        accepted_reference=_ACCEPTED_REFERENCE,
    )
    receipt = cast(
        "dict[str, object]",
        json.loads(receipt_path.read_text(encoding="utf-8")),
    )
    source_locators = cast("dict[str, list[str]]", receipt["source_locators"])
    source_files = cast(
        "dict[str, dict[str, int | str]]",
        receipt["source_files"],
    )
    upload_files = cast(
        "dict[str, dict[str, int | str]]",
        receipt["upload_files"],
    )

    assert receipt_path == (
        tmp_path / "receipts/professor/normalized-by-kaggle/v0004.json"
    )
    assert receipt["actor"] == "professor"
    assert receipt["original_kernel_id"] == "alice/pathology-job"
    assert receipt["requested_kernel_id"] == "professor/pathology-job"
    assert receipt["kernel_id"] == "professor/normalized-by-kaggle"
    assert receipt["accepted_version"] == _ACCEPTED_VERSION
    assert receipt["kernel_reference"] == _ACCEPTED_REFERENCE
    assert source_locators["dataset_sources"] == [
        "alice/patches",
        "carol/private-checkpoint",
    ]
    assert set(source_files) == {"kernel-metadata.json", "run.py"}
    assert source_files["run.py"] == upload_files["run.py"]
    assert (
        source_files["run.py"]["sha256"]
        == hashlib.sha256(
            b"print('ok')\n",
        ).hexdigest()
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_launch_receipt(
            source_dir=source,
            upload_dir=upload,
            receipt_root=tmp_path / "receipts",
            accepted_reference=_ACCEPTED_REFERENCE,
        )


def test_receipt_rejects_any_source_rewrite(tmp_path: Path) -> None:
    """A changed input owner invalidates the launch receipt."""
    source = _write_kernel(tmp_path)
    upload = tmp_path / "upload"
    create_portable_kernel_snapshot(source, upload, actor="professor")
    metadata_path = upload / "kernel-metadata.json"
    metadata = cast(
        "dict[str, object]",
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )
    metadata["dataset_sources"] = ["professor/patches"]
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed source locators"):
        write_launch_receipt(
            source_dir=source,
            upload_dir=upload,
            receipt_root=tmp_path / "receipts",
            accepted_reference="professor/pathology-job/1",
        )


def test_push_confirmation_uses_kaggles_returned_normalized_slug() -> None:
    """The accepted URL, not requested metadata, owns the result identity."""
    response = (
        "Your kernel title does not resolve to the specified id.\n"
        "Kernel version 7 successfully pushed.  Please check progress at "
        "https://www.kaggle.com/code/professor/normalized-by-kaggle\n"
    )

    reference = parse_kernel_push_confirmation(response)

    assert reference.reference == "professor/normalized-by-kaggle/7"


@pytest.mark.parametrize(
    "response",
    [
        "Kernel push error: permission denied",
        (
            "Kernel version successfully pushed. Please check progress at "
            "https://www.kaggle.com/code/professor/job"
        ),
        (
            "Kernel version 2 successfully pushed. Please check progress at "
            "https://evil.example/code/professor/job"
        ),
        (
            "The following are not valid dataset sources and could not be added "
            "to the kernel: ['other/private']\n"
            "Kernel version 3 successfully pushed.  Please check progress at "
            "https://www.kaggle.com/code/professor/job"
        ),
    ],
)
def test_push_confirmation_rejects_ambiguous_or_untrusted_success(
    response: str,
) -> None:
    """Exit-zero errors and incomplete confirmations cannot mint receipts."""
    with pytest.raises(ValueError, match=r"Kaggle|kernel"):
        parse_kernel_push_confirmation(response)


def test_download_receipt_binds_every_file_to_external_owner(tmp_path: Path) -> None:
    """Downloaded bytes retain their resource owner independently of the actor."""
    download = tmp_path / "download"
    (download / "nested").mkdir(parents=True)
    (download / "part.bin").write_bytes(b"latent-a")
    (download / "nested/manifest.csv").write_text("row\n", encoding="utf-8")

    receipt_path = write_download_receipt(
        resource_kind="dataset",
        resource_reference="external-owner/shared-latents/6",
        download_dir=download,
        receipt_name="kaggle_dataset_receipt.json",
    )
    receipt = cast(
        "dict[str, object]",
        json.loads(receipt_path.read_text(encoding="utf-8")),
    )
    files = cast("dict[str, dict[str, int | str]]", receipt["files"])

    assert receipt["resource_owner"] == "external-owner"
    assert receipt["resource_reference"] == "external-owner/shared-latents/6"
    assert set(files) == {"nested/manifest.csv", "part.bin"}
    assert (
        files["part.bin"]["sha256"]
        == hashlib.sha256(
            b"latent-a",
        ).hexdigest()
    )


def test_download_receipt_requires_exact_version(tmp_path: Path) -> None:
    """Mutable latest-version downloads are not accepted as provenance."""
    download = tmp_path / "download"
    download.mkdir()
    (download / "data.bin").write_bytes(b"data")

    with pytest.raises(ValueError, match="exact resource version"):
        write_download_receipt(
            resource_kind="dataset",
            resource_reference="owner/data",
            download_dir=download,
            receipt_name="receipt.json",
        )


def test_download_receipt_streams_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large-artifact hashing must not materialize a whole file with read_bytes."""
    download = tmp_path / "download"
    download.mkdir()
    (download / "large.bin").write_bytes(b"chunked-content")

    def reject_read_bytes(_path: pathlib.Path) -> bytes:
        message = "whole-file read_bytes is forbidden"
        raise AssertionError(message)

    monkeypatch.setattr(pathlib.Path, "read_bytes", reject_read_bytes)
    receipt_path = write_download_receipt(
        resource_kind="dataset",
        resource_reference="owner/data/2",
        download_dir=download,
        receipt_name="receipt.json",
    )

    assert receipt_path.is_file()
