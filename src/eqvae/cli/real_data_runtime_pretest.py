# Copyright 2026 HiperMaximus
"""CLI for the capped real-data runtime pretest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.real_data_runtime_pretest import (
    RealDataRuntimePretestRequest,
    write_real_data_runtime_pretest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class RealDataRuntimePretestArgs:
    """Validated CLI arguments for the capped runtime pretest."""

    config: Path
    output_dir: Path
    data_root: str | None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the capped real-data runtime pretest.

    Returns:
        Process exit code.

    """
    args = _parse_args(argv)
    write_real_data_runtime_pretest(
        RealDataRuntimePretestRequest(
            config_path=args.config,
            output_dir=args.output_dir,
            data_root=args.data_root,
        ),
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> RealDataRuntimePretestArgs:
    parser = argparse.ArgumentParser(
        description="Run the spec 0001 capped real-data runtime pretest.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    namespace = parser.parse_args(argv)
    return RealDataRuntimePretestArgs(
        config=Path(_required_str(namespace, "config")),
        output_dir=Path(_required_str(namespace, "output_dir")),
        data_root=_optional_str(namespace, "data_root"),
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


if __name__ == "__main__":
    raise SystemExit(main())
