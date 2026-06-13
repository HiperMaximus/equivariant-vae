# Copyright 2026 HiperMaximus
"""Tests for fixed-patch selector generation and validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from eqvae.cli.select_fixed_patches import main as select_fixed_patches_main
from eqvae.data.fixed_selectors import (
    DEFAULT_DATASET_SLUG,
    FIXED_25_VALIDATION_COUNT,
    FIXED_25_VALIDATION_KIND,
    FIXED_25_VALIDATION_PER_LABEL,
    FIXED_32_TRAIN_OVERFIT_COUNT,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorDocument,
    FixedSelectorGenerationContext,
    FixedSelectorRecord,
    generate_fixed_selector_document,
    load_fixed_selector_document,
    selection_key_sha256,
    validate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import (
    PATCH_SHARD_HEADER_SIZE,
    PatchRecord,
    PatchShard,
    PatchShardSpec,
)
from eqvae.data.roots import (
    TRAIN_BIN_NAME,
    TRAIN_CSV_NAME,
    VALIDATION_BIN_NAME,
    VALIDATION_CSV_NAME,
)
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

PATCH_SIZE = 8
CHANNELS = 3
LABEL_COUNT = 5
TRAIN_PATCH_COUNT = 40
VALIDATION_PATCH_COUNT = FIXED_25_VALIDATION_COUNT
VALIDATION_PATCH_COUNT_WITH_EXTRA = 30
MASKED_HOLDOUT_WSI = "synthetic_wsi_0000"
PLACEHOLDER_SELECTOR_PATH = Path("configs/spec0001/fixed_25_validation_patches.json")


def test_fixed_25_selector_canonicalizes_valid_alias(tmp_path: Path) -> None:
    """Generated selector artifacts use canonical `validation` split names."""
    bin_path, csv_path = _write_validation_shard(tmp_path)
    document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
        ),
        source_split="valid",
        context=FixedSelectorGenerationContext(data_root=tmp_path),
    )

    validated = validate_fixed_selector_document(
        document=document,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
        ),
        expected_kind=FIXED_25_VALIDATION_KIND,
    )
    label_counts = Counter(selector.label for selector in validated)

    assert document.source_split == "validation"
    assert document.source.source_split == "validation"
    assert len(validated) == FIXED_25_VALIDATION_COUNT
    expected_label_counts = dict.fromkeys(
        range(LABEL_COUNT),
        FIXED_25_VALIDATION_PER_LABEL,
    )
    assert dict(sorted(label_counts.items())) == expected_label_counts
    assert all(
        selector.sample_id.startswith("validation:") for selector in document.selectors
    )
    assert document.source.data_root == tmp_path


def test_fixed_32_train_selector_excludes_masked_holdout(tmp_path: Path) -> None:
    """Tiny-overfit selectors never draw from the masked holdout WSI list."""
    bin_path, csv_path = _write_train_shard(tmp_path)
    masked_holdout_wsi_ids = frozenset({MASKED_HOLDOUT_WSI})
    document = generate_fixed_selector_document(
        selector_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
        ),
        source_split="train",
        context=FixedSelectorGenerationContext(
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        ),
    )
    validated = validate_fixed_selector_document(
        document=document,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
        ),
        expected_kind=FIXED_32_TRAIN_OVERFIT_KIND,
        masked_holdout_wsi_ids=masked_holdout_wsi_ids,
    )

    assert len(validated) == FIXED_32_TRAIN_OVERFIT_COUNT
    assert {selector.wsi_id for selector in document.selectors}.isdisjoint(
        {MASKED_HOLDOUT_WSI},
    )


def test_placeholder_selector_configs_are_rejected() -> None:
    """Canonical placeholder configs cannot be consumed as real selectors."""
    with pytest.raises(ValueError, match="not ready"):
        load_fixed_selector_document(PLACEHOLDER_SELECTOR_PATH)


def test_canonical_selector_write_requires_overwrite_and_crc(
    tmp_path: Path,
) -> None:
    """Tracked selector configs need explicit approval plus CRC evidence."""
    canonical_path = (
        tmp_path / "configs" / "spec0001" / ("fixed_25_validation_patches.json")
    )
    bin_path, csv_path = _write_validation_shard(tmp_path)
    fast_document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
            validate_crc=False,
        ),
        source_split="validation",
    )

    with pytest.raises(PermissionError, match="allow-tracked-config-overwrite"):
        write_fixed_selector_document(path=canonical_path, document=fast_document)
    with pytest.raises(PermissionError, match="CRC-validated"):
        write_fixed_selector_document(
            path=canonical_path,
            document=fast_document,
            allow_tracked_config_overwrite=True,
        )

    crc_document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=PatchShardSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            image_size=PATCH_SIZE,
            validate_crc=True,
        ),
        source_split="validation",
    )
    write_fixed_selector_document(
        path=canonical_path,
        document=crc_document,
        allow_tracked_config_overwrite=True,
    )

    assert canonical_path.exists()


def test_selector_validation_rejects_noncanonical_but_valid_row(
    tmp_path: Path,
) -> None:
    """Validation recomputes the canonical selection policy."""
    bin_path, csv_path = _write_validation_shard(
        tmp_path,
        count=VALIDATION_PATCH_COUNT_WITH_EXTRA,
    )
    shard_spec = PatchShardSpec(
        bin_path=bin_path,
        csv_path=csv_path,
        image_size=PATCH_SIZE,
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=shard_spec,
        source_split="validation",
    )
    shard = PatchShard(shard_spec)
    selected_file_indices = {selector.file_index for selector in document.selectors}
    replacement = next(
        record
        for record in shard.records
        if record.label == document.selectors[0].label
        and record.file_index not in selected_file_indices
    )
    wrong_selector = _selector_for_record(
        rank=0,
        record=replacement,
        document=document,
        bin_path=bin_path,
        patch_bytes=CHANNELS * PATCH_SIZE * PATCH_SIZE,
    )
    wrong_document = replace(
        document,
        selectors=(wrong_selector, *document.selectors[1:]),
    )

    with pytest.raises(ValueError, match="Selector policy mismatch"):
        validate_fixed_selector_document(
            document=wrong_document,
            shard_spec=shard_spec,
            expected_kind=FIXED_25_VALIDATION_KIND,
        )


def test_selector_validation_rejects_seed_and_csv_drift(tmp_path: Path) -> None:
    """Selector JSON is audit evidence that is replayed against current data."""
    bin_path, csv_path = _write_validation_shard(tmp_path)
    shard_spec = PatchShardSpec(
        bin_path=bin_path,
        csv_path=csv_path,
        image_size=PATCH_SIZE,
    )
    document = generate_fixed_selector_document(
        selector_kind=FIXED_25_VALIDATION_KIND,
        shard_spec=shard_spec,
        source_split="validation",
    )
    seed_tampered = replace(document, selector_seed="wrong-seed")

    with pytest.raises(ValueError, match="Selector seed"):
        validate_fixed_selector_document(
            document=seed_tampered,
            shard_spec=shard_spec,
            expected_kind=FIXED_25_VALIDATION_KIND,
        )

    _rewrite_first_wsi_id(csv_path, replacement="synthetic_wsi_drifted")
    with pytest.raises(ValueError, match="csv_sha256"):
        validate_fixed_selector_document(
            document=document,
            shard_spec=shard_spec,
            expected_kind=FIXED_25_VALIDATION_KIND,
        )


def test_selector_generation_rejects_duplicate_semantic_identity(
    tmp_path: Path,
) -> None:
    """Duplicate `wsi_id,label,x,y` rows make deterministic selection unsafe."""
    bin_path, csv_path = _write_validation_shard(tmp_path)
    _duplicate_first_semantic_identity(csv_path)

    with pytest.raises(ValueError, match="duplicate semantic identities"):
        generate_fixed_selector_document(
            selector_kind=FIXED_25_VALIDATION_KIND,
            shard_spec=PatchShardSpec(
                bin_path=bin_path,
                csv_path=csv_path,
                image_size=PATCH_SIZE,
            ),
            source_split="validation",
        )


def test_select_fixed_patches_cli_writes_noncanonical_synthetic_output(
    tmp_path: Path,
) -> None:
    """Local synthetic selector generation writes under ignored run paths."""
    root = tmp_path / "synthetic-root"
    output = tmp_path / "runs" / "fixed_25_validation_patches.json"
    _write_complete_data_root(root)

    exit_code = select_fixed_patches_main(
        [
            "--config",
            "configs/spec0001/non_eq_vae_kaggle_debug.json",
            "--data-root",
            str(root),
            "--output",
            str(output),
            "--kind",
            FIXED_25_VALIDATION_KIND,
            "--image-size",
            str(PATCH_SIZE),
            "--channels",
            str(CHANNELS),
        ],
    )
    payload = _load_json(output)
    source = _object_field(payload, "source")

    assert exit_code == 0
    assert payload["status"] == "pass"
    assert payload["selector_kind"] == FIXED_25_VALIDATION_KIND
    assert payload["source_split"] == "validation"
    assert source["dataset_slug"] == DEFAULT_DATASET_SLUG
    assert source["data_root"] == str(root / "dataset")
    assert len(_list_field(payload, "selectors")) == FIXED_25_VALIDATION_COUNT


def test_select_fixed_patches_cli_rejects_nonpositive_shape(
    tmp_path: Path,
) -> None:
    """CLI shape overrides fail instead of falling through to defaults."""
    root = tmp_path / "synthetic-root"
    output = tmp_path / "runs" / "fixed_25_validation_patches.json"
    _write_complete_data_root(root)

    with pytest.raises(ValueError, match="image_size must be positive"):
        select_fixed_patches_main(
            [
                "--config",
                "configs/spec0001/non_eq_vae_kaggle_debug.json",
                "--data-root",
                str(root),
                "--output",
                str(output),
                "--kind",
                FIXED_25_VALIDATION_KIND,
                "--image-size",
                "0",
                "--channels",
                str(CHANNELS),
            ],
        )


def _write_train_shard(root: Path) -> tuple[Path, Path]:
    bin_path = root / TRAIN_BIN_NAME
    csv_path = root / TRAIN_CSV_NAME
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=TRAIN_PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )
    return bin_path, csv_path


def _write_validation_shard(
    root: Path,
    *,
    count: int = VALIDATION_PATCH_COUNT,
) -> tuple[Path, Path]:
    bin_path = root / VALIDATION_BIN_NAME
    csv_path = root / VALIDATION_CSV_NAME
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(count=count, image_size=PATCH_SIZE),
        include_idx=True,
    )
    return bin_path, csv_path


def _write_complete_data_root(root: Path) -> None:
    dataset = root / "dataset"
    write_synthetic_patch_shard(
        bin_path=dataset / TRAIN_BIN_NAME,
        csv_path=dataset / TRAIN_CSV_NAME,
        spec=SyntheticPatchSpec(count=TRAIN_PATCH_COUNT, image_size=PATCH_SIZE),
        include_idx=False,
    )
    write_synthetic_patch_shard(
        bin_path=dataset / VALIDATION_BIN_NAME,
        csv_path=dataset / VALIDATION_CSV_NAME,
        spec=SyntheticPatchSpec(
            count=VALIDATION_PATCH_COUNT,
            image_size=PATCH_SIZE,
        ),
        include_idx=True,
    )


def _selector_for_record(
    *,
    rank: int,
    record: PatchRecord,
    document: FixedSelectorDocument,
    bin_path: Path,
    patch_bytes: int,
) -> FixedSelectorRecord:
    return FixedSelectorRecord(
        rank=rank,
        source_split=document.source_split,
        file_index=record.file_index,
        row_index=record.row_index,
        sample_id=record.sample_id(document.source_split),
        wsi_id=record.wsi_id,
        label=record.label,
        x=record.x,
        y=record.y,
        selection_key_sha256=selection_key_sha256(
            seed=document.selector_seed,
            record=record,
        ),
        patch_sha256=_patch_sha256(
            bin_path=bin_path,
            file_index=record.file_index,
            patch_bytes=patch_bytes,
        ),
    )


def _patch_sha256(*, bin_path: Path, file_index: int, patch_bytes: int) -> str:
    with bin_path.open("rb") as binary_file:
        binary_file.seek(PATCH_SHARD_HEADER_SIZE + file_index * patch_bytes)
        payload = binary_file.read(patch_bytes)
    return hashlib.sha256(payload).hexdigest()


def _rewrite_first_wsi_id(csv_path: Path, *, replacement: str) -> None:
    rows = _read_csv_rows(csv_path)
    rows[1]["wsi_id"] = replacement
    _write_csv_rows(csv_path, rows)


def _duplicate_first_semantic_identity(csv_path: Path) -> None:
    rows = _read_csv_rows(csv_path)
    first = rows[1]
    rows[2]["wsi_id"] = first["wsi_id"]
    rows[2]["label"] = first["label"]
    rows[2]["x"] = first["x"]
    rows[2]["y"] = first["y"]
    _write_csv_rows(csv_path, rows)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_csv_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = tuple(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, object]:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        message = f"Expected JSON object in {path}"
        raise TypeError(message)
    return cast("dict[str, object]", payload)


def _object_field(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"Expected object field {key!r}"
        raise TypeError(message)
    return cast("dict[str, object]", value)


def _list_field(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected list field {key!r}"
        raise TypeError(message)
    return cast("list[object]", value)
