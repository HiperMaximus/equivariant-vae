# Copyright 2026 HiperMaximus
"""CLI for writing the spec 0001 model-count artifact."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.model_count import write_model_count

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ModelCountArgs:
    """Validated arguments for the model-count CLI."""

    config: Path
    output: Path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the spec 0001 `benchmark/model_count.json` artifact.

    Returns:
        Process exit status.

    """
    args = _parse_args(argv)
    write_model_count(config_path=args.config, output_path=args.output)
    return 0


def _parse_args(argv: Sequence[str] | None) -> ModelCountArgs:
    parser = argparse.ArgumentParser(
        description="Write the spec 0001 model-count artifact.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    namespace = parser.parse_args(argv)
    return ModelCountArgs(
        config=Path(_required_str(namespace, "config")),
        output=Path(_required_str(namespace, "output")),
    )


def _required_str(namespace: argparse.Namespace, name: str) -> str:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, str):
        return value
    message = f"Expected string argument: {name}"
    raise TypeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
