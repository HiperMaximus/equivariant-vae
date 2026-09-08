# Copyright 2026 HiperMaximus
"""Build the immutable local Spec 0021 Kaggle input dataset directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from eqvae.data.latent_shards import EXPECTED_TASK_MANIFEST_SHA256
from eqvae.inference.input_bundle import stage_fresh_input_bundle

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_DATASET_SLUG: Final = "maximusshtefan/eqvae-ubc-ocean-latent-inputs"
_DEFAULT_MANIFEST_ROOT: Final = Path("runs/local/ubc_ocean_eval_consumption")
_DEFAULT_NORMAL_CHECKPOINT: Final = Path(
    "runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt",
)
_DEFAULT_SO2_CHECKPOINT: Final = Path(
    "runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/"
    "checkpoints/step_060000.pt",
)


class _Arguments(argparse.Namespace):
    output: Path
    manifest_root: Path
    normal_checkpoint: Path
    so2_checkpoint: Path
    dataset_slug: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=_DEFAULT_MANIFEST_ROOT,
        help="Spec 0019 consumption directory (default: %(default)s).",
    )
    parser.add_argument(
        "--normal-checkpoint",
        type=Path,
        default=_DEFAULT_NORMAL_CHECKPOINT,
    )
    parser.add_argument(
        "--so2-checkpoint",
        type=Path,
        default=_DEFAULT_SO2_CHECKPOINT,
    )
    parser.add_argument("--dataset-slug", default=_DEFAULT_DATASET_SLUG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Stage and revalidate the exact local upload directory.

    Returns:
        Zero after the complete contract has been durably written.

    """
    arguments = cast("_Arguments", _parser().parse_args(argv))
    manifest_root = arguments.manifest_root
    validated = stage_fresh_input_bundle(
        arguments.output,
        work_manifests={
            run: manifest_root / "work_shards" / f"run_{run:02d}_of_05.csv"
            for run in range(1, 6)
        },
        union_manifest=manifest_root / "union_patch_manifest.csv",
        task_manifests={
            name: manifest_root / f"{name}.csv"
            for name in EXPECTED_TASK_MANIFEST_SHA256
        },
        normal_checkpoint=arguments.normal_checkpoint,
        so2_checkpoint=arguments.so2_checkpoint,
        dataset_slug=arguments.dataset_slug,
    )
    summary = {
        "dataset_slug": validated.dataset_slug,
        "files": len(validated.files),
        "provenance_sha256": validated.provenance_sha256,
        "root": str(validated.root),
    }
    sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
