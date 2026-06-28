# Copyright 2026 HiperMaximus
"""CLI for the selected-runtime real UBC train runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from eqvae.training.selected_runtime_runner import (
    SelectedRuntimeTrainRequest,
    write_selected_runtime_training_run,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _ParsedArgs(Protocol):
    config: Path
    runtime_config: Path
    data: str
    data_root: str | None
    fixed_train_patches: Path | None
    output_dir: Path
    run_name: str
    resume: Path | None
    max_train_steps: int | None
    max_val_steps: int | None
    save_every_steps: int | None
    dry_run: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected-runtime trainer from command-line arguments.

    Returns:
        Process-style exit code.

    """
    args = cast("_ParsedArgs", _parser().parse_args(argv))
    write_selected_runtime_training_run(
        SelectedRuntimeTrainRequest(
            config_path=args.config,
            runtime_config=args.runtime_config,
            data=args.data,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
            output_dir=args.output_dir,
            run_name=args.run_name,
            resume=args.resume,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            save_every_steps=args.save_every_steps,
            dry_run=args.dry_run,
        ),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the selected-runtime real UBC training proof.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument(
        "--data",
        choices=("synthetic", "ubc-pre-shuffled"),
        required=True,
    )
    parser.add_argument("--data-root")
    parser.add_argument("--fixed-train-patches", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--save-every-steps", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use local non-promotable execution while writing runner artifacts.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
