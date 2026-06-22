# Copyright 2026 HiperMaximus
"""CLI for short spec 0001 debug/tiny training proof runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.training.debug import DebugTrainingRequest, write_debug_training_run

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class TrainArgs:
    """Validated CLI arguments for debug/tiny training proof runs."""

    config: Path
    data: str
    output_dir: Path
    run_name: str
    runtime_config: Path | None
    data_root: str | None
    fixed_train_patches: Path | None
    resume: Path | None
    max_train_steps: int | None
    max_val_steps: int | None
    save_every_steps: int | None


def main(argv: Sequence[str] | None = None) -> int:
    """Run a short debug/tiny training proof.

    Returns:
        Process exit status.

    """
    args = _parse_args(argv)
    write_debug_training_run(
        DebugTrainingRequest(
            config_path=args.config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            data=args.data,
            runtime_config=args.runtime_config,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
            resume=args.resume,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
            save_every_steps=args.save_every_steps,
        ),
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="Run spec 0001 short debug/tiny training proof.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-config")
    parser.add_argument(
        "--data",
        choices=("synthetic", "ubc-pre-shuffled"),
        required=True,
    )
    parser.add_argument("--data-root")
    parser.add_argument("--fixed-train-patches")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--save-every-steps", type=int)
    namespace = parser.parse_args(argv)
    runtime_config = _optional_path(namespace, "runtime_config")
    fixed_train_patches = _optional_path(namespace, "fixed_train_patches")
    resume = _optional_path(namespace, "resume")
    return TrainArgs(
        config=Path(_required_str(namespace, "config")),
        data=_required_str(namespace, "data"),
        output_dir=Path(_required_str(namespace, "output_dir")),
        run_name=_required_str(namespace, "run_name"),
        runtime_config=runtime_config,
        data_root=_optional_str(namespace, "data_root"),
        fixed_train_patches=fixed_train_patches,
        resume=resume,
        max_train_steps=_optional_int(namespace, "max_train_steps"),
        max_val_steps=_optional_int(namespace, "max_val_steps"),
        save_every_steps=_optional_int(namespace, "save_every_steps"),
    )


def _required_str(namespace: argparse.Namespace, name: str) -> str:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, str):
        return value
    message = f"Expected string argument: {name}"
    raise TypeError(message)


def _optional_str(namespace: argparse.Namespace, name: str) -> str | None:
    value = cast("object", getattr(namespace, name))
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string argument: {name}"
    raise TypeError(message)


def _optional_int(namespace: argparse.Namespace, name: str) -> int | None:
    value = cast("object", getattr(namespace, name))
    if value is None or isinstance(value, int):
        return value
    message = f"Expected optional integer argument: {name}"
    raise TypeError(message)


def _optional_path(namespace: argparse.Namespace, name: str) -> Path | None:
    value = _optional_str(namespace, name)
    return None if value is None else Path(value)


if __name__ == "__main__":
    raise SystemExit(main())
