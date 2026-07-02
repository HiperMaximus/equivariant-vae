# Copyright 2026 HiperMaximus
"""Save the fixed-25 selected patch images without a trained model.

Companion to ``eqvae.cli.select_fixed_patches``: given a ready fixed-25 selector
and the validation shard, this loads the 25 canonical validation patches (failing
closed on the placeholder / any noncanonical selector) and writes the immutable
originals archive (``artifacts/fixed25/originals.pt`` plus an ``originals.png``
montage) via the shared Spec 0010 ``write_originals``. It needs no checkpoint, so
it runs alongside selector generation (e.g. inside the Kaggle selector kernel where
the real UBC shard is mounted) to persist *which* patches were selected AND the
patches themselves for inspection before the tracked selector config is committed.

This is inspection tooling only; it never touches training or the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.artifacts.fixed25_equivariance import (
    FIXED25_DIRNAME,
    ORIGINALS_PNG,
    ORIGINALS_PT,
    load_fixed25_patches,
    parse_fixed25_config,
    validation_shard_spec_for,
    write_originals,
)
from eqvae.config import resolve_json_config
from eqvae.data.roots import resolve_patch_data_paths
from eqvae.data.training_batches import PatchTrainingDataset, PatchTrainingDatasetSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.artifacts.fixed25_equivariance import Fixed25Config, Fixed25Patches
    from eqvae.benchmarking.io import JsonObject

_DEFAULT_IMAGE_SIZE = 256
_DEFAULT_LATENT_SEED = 20260612
_DEFAULT_CHANNELS = 3


@dataclass(frozen=True)
class Fixed25OriginalsArgs:
    """Validated arguments for the fixed-25 originals archiver."""

    config: Path
    output_dir: Path
    data_root: str
    selector: Path | None


def main(argv: Sequence[str] | None = None) -> int:
    """Load the fixed-25 patches and write the originals archive.

    Returns:
        Process exit status.

    Raises:
        ValueError: If the resolved config has no ``fixed25_equivariance`` block.

    """
    args = _parse_args(argv)
    resolved = resolve_json_config(args.config)
    effective = resolved.effective_config
    config = parse_fixed25_config(
        effective,
        default_epsilon_seed=_int_field(
            _object_field(effective, "seeds"),
            "latent_seed",
            default=_DEFAULT_LATENT_SEED,
        ),
    )
    if config is None:
        message = f"config {args.config} has no fixed25_equivariance block"
        raise ValueError(message)
    image_size = _int_field(
        _object_field(effective, "data"),
        "image_size",
        default=_DEFAULT_IMAGE_SIZE,
    )
    validation_dataset = _validation_dataset(
        data_root=args.data_root,
        image_size=image_size,
    )
    try:
        patches = _load_patches(
            args=args,
            config=config,
            validation_dataset=validation_dataset,
            image_size=image_size,
        )
    finally:
        validation_dataset.close()
    fixed25_dir = args.output_dir / "artifacts" / FIXED25_DIRNAME
    write_originals(fixed25_dir=fixed25_dir, patches=patches)
    _report(patches=patches, fixed25_dir=fixed25_dir)
    return 0


def _load_patches(
    *,
    args: Fixed25OriginalsArgs,
    config: Fixed25Config,
    validation_dataset: PatchTrainingDataset,
    image_size: int,
) -> Fixed25Patches:
    paths = resolve_patch_data_paths(args.data_root).validation
    shard_spec = validation_shard_spec_for(
        validation_bin_path=paths.bin_path,
        validation_csv_path=paths.csv_path,
        image_size=image_size,
        # The canonical selector is CRC-validated, so validate here with CRC too;
        # validate_fixed_selector_document compares crc_checked for equality.
        validate_crc=True,
    )
    selector_path = args.selector or Path(config.selector_config)
    return load_fixed25_patches(
        config=config,
        selector_path=selector_path,
        validation_shard_spec=shard_spec,
        validation_dataset=validation_dataset,
    )


def _report(*, patches: Fixed25Patches, fixed25_dir: Path) -> None:
    labels = sorted(
        int(cast("int", identity["label"])) for identity in patches.identities
    )
    print(  # noqa: T201 - Kaggle logs need an explicit selected-patch summary.
        f"fixed-25 originals: wrote {patches.images.shape[0]} patches "
        f"(labels {labels}) to {fixed25_dir / ORIGINALS_PT} and "
        f"{fixed25_dir / ORIGINALS_PNG}; selector sha256 {patches.selector_sha256}",
        flush=True,
    )


def _validation_dataset(
    *,
    data_root: str,
    image_size: int,
) -> PatchTrainingDataset:
    paths = resolve_patch_data_paths(data_root).validation
    return PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.bin_path,
            csv_path=paths.csv_path,
            split="validation",
            image_size=image_size,
            channels=_DEFAULT_CHANNELS,
            validate_crc=False,
        ),
    )


def _object_field(obj: JsonObject, key: str) -> JsonObject:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def _int_field(obj: JsonObject, key: str, *, default: int) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _parse_args(argv: Sequence[str] | None) -> Fixed25OriginalsArgs:
    parser = argparse.ArgumentParser(
        description="Write the fixed-25 selected patch originals (no model needed).",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default="auto")
    parser.add_argument("--selector", type=Path, default=None)
    namespace = parser.parse_args(argv)
    return Fixed25OriginalsArgs(
        config=cast("Path", namespace.config),
        output_dir=cast("Path", namespace.output_dir),
        data_root=cast("str", namespace.data_root),
        selector=cast("Path | None", namespace.selector),
    )


if __name__ == "__main__":
    raise SystemExit(main())
