# Copyright 2026 HiperMaximus
"""CLI for generating spec 0001 fixed-patch selector artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.config import JsonObject, JsonValue, resolve_json_config
from eqvae.data.fixed_selectors import (
    DEFAULT_DATASET_SLUG,
    FIXED_25_VALIDATION_KIND,
    FIXED_32_TRAIN_OVERFIT_KIND,
    FixedSelectorGenerationContext,
    SelectorKind,
    generate_fixed_selector_document,
    infer_selector_kind,
    validate_fixed_selector_document,
    write_fixed_selector_document,
)
from eqvae.data.patch_shards import PatchShardSpec
from eqvae.data.roots import resolve_patch_data_paths
from eqvae.data.splits import load_masked_holdout_wsi_ids

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MASKED_HOLDOUT_CSV = (
    REPO_ROOT / "docs" / "data" / ("ubc_ocean_masked_holdout_ids.csv")
)
DEFAULT_IMAGE_SIZE = 256
DEFAULT_CHANNELS = 3


@dataclass(frozen=True)
class SelectFixedPatchesArgs:
    """Validated arguments for fixed selector generation."""

    config: Path
    data_root: str | None
    output: Path
    kind: str | None
    image_size: int | None
    channels: int | None
    validate_crc: bool
    allow_tracked_config_overwrite: bool
    masked_holdout_csv: Path


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and validate one fixed selector artifact.

    Returns:
        Process exit status.

    """
    args = _parse_args(argv)
    resolved = resolve_json_config(args.config)
    data_config = _object_field(resolved.effective_config, "data")
    selector_kind = infer_selector_kind(args.output, explicit_kind=args.kind)
    split = "validation" if selector_kind == FIXED_25_VALIDATION_KIND else "train"
    data_root = args.data_root or _optional_str(data_config, "data_root") or "auto"
    paths = resolve_patch_data_paths(data_root)
    split_paths = paths.for_split(split)
    image_size = _first_optional_int(
        args.image_size,
        _optional_int(data_config, "image_size"),
    )
    channels = _first_optional_int(
        args.channels,
        _optional_int(data_config, "channels"),
    )
    resolved_image_size = _resolve_optional_int(image_size, DEFAULT_IMAGE_SIZE)
    resolved_channels = _resolve_optional_int(channels, DEFAULT_CHANNELS)
    _validate_positive("image_size", resolved_image_size)
    _validate_positive("channels", resolved_channels)
    masked_holdout_wsi_ids = _masked_holdout_ids(
        selector_kind=selector_kind,
        csv_path=args.masked_holdout_csv,
    )
    shard_spec = PatchShardSpec(
        bin_path=split_paths.bin_path,
        csv_path=split_paths.csv_path,
        image_size=resolved_image_size,
        channels=resolved_channels,
        validate_crc=args.validate_crc,
    )
    document = generate_fixed_selector_document(
        selector_kind=selector_kind,
        shard_spec=shard_spec,
        source_split=split,
        context=FixedSelectorGenerationContext(
            dataset_slug=_optional_str(data_config, "dataset_slug")
            or DEFAULT_DATASET_SLUG,
            data_root=paths.root,
            masked_holdout_wsi_ids=masked_holdout_wsi_ids,
        ),
    )
    validate_fixed_selector_document(
        document=document,
        shard_spec=shard_spec,
        expected_kind=selector_kind,
        masked_holdout_wsi_ids=masked_holdout_wsi_ids,
    )
    write_fixed_selector_document(
        path=args.output,
        document=document,
        allow_tracked_config_overwrite=args.allow_tracked_config_overwrite,
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> SelectFixedPatchesArgs:
    parser = argparse.ArgumentParser(
        description="Generate spec 0001 fixed selector JSON.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--kind",
        choices=(FIXED_25_VALIDATION_KIND, FIXED_32_TRAIN_OVERFIT_KIND),
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--validate-crc", action="store_true")
    parser.add_argument("--allow-tracked-config-overwrite", action="store_true")
    parser.add_argument(
        "--masked-holdout-csv",
        default=str(DEFAULT_MASKED_HOLDOUT_CSV),
    )
    namespace = parser.parse_args(argv)
    return SelectFixedPatchesArgs(
        config=Path(_required_str(namespace, "config")),
        data_root=_optional_namespace_str(namespace, "data_root"),
        output=Path(_required_str(namespace, "output")),
        kind=_optional_namespace_str(namespace, "kind"),
        image_size=_optional_namespace_int(namespace, "image_size"),
        channels=_optional_namespace_int(namespace, "channels"),
        validate_crc=_required_bool(namespace, "validate_crc"),
        allow_tracked_config_overwrite=_required_bool(
            namespace,
            "allow_tracked_config_overwrite",
        ),
        masked_holdout_csv=Path(_required_str(namespace, "masked_holdout_csv")),
    )


def _masked_holdout_ids(
    *,
    selector_kind: SelectorKind,
    csv_path: Path,
) -> frozenset[str]:
    if selector_kind != FIXED_32_TRAIN_OVERFIT_KIND:
        return frozenset()
    return load_masked_holdout_wsi_ids(csv_path)


def _object_field(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if not isinstance(value, dict):
        message = f"Expected object config field {key!r}"
        raise TypeError(message)
    return cast("JsonObject", value)


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = _optional_value(payload, key)
    if value is None:
        return None
    if not isinstance(value, str):
        message = f"Expected string config field {key!r}"
        raise TypeError(message)
    return value


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = _optional_value(payload, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"Expected integer config field {key!r}"
        raise TypeError(message)
    return value


def _optional_value(payload: JsonObject, key: str) -> JsonValue | None:
    return payload.get(key)


def _first_optional_int(first: int | None, second: int | None) -> int | None:
    return first if first is not None else second


def _resolve_optional_int(value: int | None, default: int) -> int:
    return default if value is None else value


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        message = f"{name} must be positive, got {value}"
        raise ValueError(message)


def _required_str(namespace: argparse.Namespace, name: str) -> str:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, str):
        return value
    message = f"Expected string argument: {name}"
    raise TypeError(message)


def _optional_namespace_str(
    namespace: argparse.Namespace,
    name: str,
) -> str | None:
    value = cast("object", getattr(namespace, name))
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string argument: {name}"
    raise TypeError(message)


def _optional_namespace_int(
    namespace: argparse.Namespace,
    name: str,
) -> int | None:
    value = cast("object", getattr(namespace, name))
    if value is None or isinstance(value, int):
        return value
    message = f"Expected optional integer argument: {name}"
    raise TypeError(message)


def _required_bool(namespace: argparse.Namespace, name: str) -> bool:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, bool):
        return value
    message = f"Expected boolean argument: {name}"
    raise TypeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
