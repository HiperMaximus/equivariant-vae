# Copyright 2026 HiperMaximus
"""CLI for Spec 0008 fixed-32 selector readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from eqvae.benchmarking.fixed32_selector_readiness import (
    Fixed32RemoteGenerateReadinessRequest,
    write_fixed32_remote_generate_readiness,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _ParsedArgs(Protocol):
    config: Path
    synthetic_root: Path
    output_dir: Path
    masked_holdout_csv: Path
    image_size: int
    channels: int


def main(argv: Sequence[str] | None = None) -> int:
    """Write the local fixed-32 remote-generate readiness artifact.

    Returns:
        Process-style exit code.

    """
    args = cast("_ParsedArgs", _parser().parse_args(argv))
    write_fixed32_remote_generate_readiness(
        Fixed32RemoteGenerateReadinessRequest(
            output_dir=args.output_dir,
            synthetic_root=args.synthetic_root,
            config_path=args.config,
            masked_holdout_csv=args.masked_holdout_csv,
            image_size=args.image_size,
            channels=args.channels,
        ),
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local Spec 0008 fixed-32 selector readiness.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--masked-holdout-csv", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
