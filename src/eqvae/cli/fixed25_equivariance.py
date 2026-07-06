# Copyright 2026 HiperMaximus
"""Standalone Spec 0010 fixed-25 embedding-equivariance evaluator.

Re-runs the fixed-25 protocol over any saved checkpoint (``best_model.pt`` /
``final.pt``) and, once it exists, over the future ``SO(2)``-steerable model, using
byte-identical evaluation config and the same frozen 25 validation patches. This is
an evaluation / inspection tool only; it never touches training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from eqvae.artifacts.fixed25_equivariance import (
    FIXED25_DIRNAME,
    compute_rot90_exactness,
    evaluate_boundary,
    load_fixed25_patches,
    parse_fixed25_config,
    validation_shard_spec_for,
    write_equivariance_csv,
    write_manifest,
    write_originals,
)
from eqvae.config import resolve_json_config
from eqvae.data.roots import resolve_patch_data_paths
from eqvae.data.training_batches import PatchTrainingDataset, PatchTrainingDatasetSpec
from eqvae.models.non_equivariant_vae import DEFAULT_GROUPNORM_GROUPS
from eqvae.models.registry import MODEL_KIND_NON_EQ_TRANSLATABLE, build_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eqvae.artifacts.fixed25_equivariance import Fixed25Config
    from eqvae.benchmarking.io import JsonObject
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE

_DEFAULT_IMAGE_SIZE = 256
_DEFAULT_LATENT_SEED = 20260612


@dataclass(frozen=True)
class Fixed25StandaloneArgs:
    """Validated arguments for the standalone fixed-25 evaluator."""

    config: Path
    checkpoint: Path
    output_dir: Path
    data_root: str
    selector: Path | None
    optimizer_step: int
    promotable: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone fixed-25 evaluation for one checkpoint.

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
    norm_groups = _int_field(
        _object_field(_object_field(effective, "model"), "normalization"),
        "num_groups",
        default=DEFAULT_GROUPNORM_GROUPS,
    )
    validation_dataset = _validation_dataset(
        data_root=args.data_root,
        image_size=image_size,
    )
    try:
        _run(
            args=args,
            config=config,
            validation_dataset=validation_dataset,
            image_size=image_size,
            norm_groups=norm_groups,
        )
    finally:
        validation_dataset.close()
    return 0


def _run(
    *,
    args: Fixed25StandaloneArgs,
    config: Fixed25Config,
    validation_dataset: PatchTrainingDataset,
    image_size: int,
    norm_groups: int,
) -> None:
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
    patches = load_fixed25_patches(
        config=config,
        selector_path=selector_path,
        validation_shard_spec=shard_spec,
        validation_dataset=validation_dataset,
    )
    model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": norm_groups},
    )
    _load_model_weights(model=model, checkpoint=args.checkpoint)
    model.eval()
    device = torch.device("cpu")
    # Safe default: artifacts are non-promotable unless the operator explicitly
    # opts in with --promotable (which asserts a real, canonical evaluation), so a
    # forgotten flag can never mislabel synthetic output as issue evidence.
    data_source = "real" if args.promotable else "synthetic"
    promotable = args.promotable
    fixed25_dir = args.output_dir / "artifacts" / FIXED25_DIRNAME
    exactness = compute_rot90_exactness(model=model, patches=patches, device=device)
    write_originals(fixed25_dir=fixed25_dir, patches=patches)
    rows = evaluate_boundary(
        model=model,
        patches=patches,
        config=config,
        fixed25_dir=fixed25_dir,
        optimizer_step=args.optimizer_step,
        device=device,
        data_source=data_source,
        promotable=promotable,
    )
    write_equivariance_csv(
        path=args.output_dir / "metrics" / "equivariance_25.csv",
        rows=rows,
    )
    write_manifest(
        fixed25_dir=fixed25_dir,
        config=config,
        patches=patches,
        data_source=data_source,
        promotable=promotable,
        rot90_exactness_error=exactness,
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
            channels=3,
            validate_crc=False,
        ),
    )


def _load_model_weights(*, model: NonEquivariantVAE, checkpoint: Path) -> None:
    payload = cast(
        "object",
        torch.load(checkpoint, map_location="cpu", weights_only=False),
    )
    state: object = payload
    if isinstance(payload, dict):
        payload_dict = cast("dict[str, object]", payload)
        if "model_state_dict" in payload_dict:
            state = payload_dict["model_state_dict"]
    if not isinstance(state, dict):
        message = f"checkpoint {checkpoint} does not contain a model state dict"
        raise TypeError(message)
    model.load_state_dict(cast("dict[str, object]", state))


def _object_field(obj: JsonObject, key: str) -> JsonObject:
    value = obj.get(key)
    return value if isinstance(value, dict) else {}


def _int_field(obj: JsonObject, key: str, *, default: int) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _parse_args(argv: Sequence[str] | None) -> Fixed25StandaloneArgs:
    parser = argparse.ArgumentParser(
        description="Run the Spec 0010 fixed-25 embedding-equivariance evaluator.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=str, default="auto")
    parser.add_argument("--selector", type=Path, default=None)
    parser.add_argument("--optimizer-step", type=int, default=0)
    parser.add_argument(
        "--promotable",
        action="store_true",
        help=(
            "Opt in to labeling artifacts real / promotable; omit for the safe "
            "synthetic / non-promotable default."
        ),
    )
    namespace = parser.parse_args(argv)
    return Fixed25StandaloneArgs(
        config=cast("Path", namespace.config),
        checkpoint=cast("Path", namespace.checkpoint),
        output_dir=cast("Path", namespace.output_dir),
        data_root=cast("str", namespace.data_root),
        selector=cast("Path | None", namespace.selector),
        optimizer_step=cast("int", namespace.optimizer_step),
        promotable=cast("bool", namespace.promotable),
    )


if __name__ == "__main__":
    raise SystemExit(main())
