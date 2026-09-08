# pyright: reportAny=false
# Copyright 2026 HiperMaximus
# ruff: noqa: T201
"""Render the local Spec 0044 professor-requested metrics package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from eqvae.evaluation.professor_metrics import (
    DEFAULT_INPUT_CONTRACT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMOOTHING_WINDOW,
    build_professor_metrics_package,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Build the hash-checked local evidence package.

    Returns:
        Zero on success.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", type=Path, default=DEFAULT_INPUT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Odd centered moving-average window used for train traces only.",
    )
    args = parser.parse_args(argv)
    output = build_professor_metrics_package(
        input_contract=args.input_contract,
        output_dir=args.output_dir,
        smoothing_window=args.smoothing_window,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
