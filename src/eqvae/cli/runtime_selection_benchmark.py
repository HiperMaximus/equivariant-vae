# Copyright 2026 HiperMaximus
"""CLI for the fail-closed selected-runtime benchmark artifact path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.runtime_selection import (
    RuntimeSelectionBenchmarkRequest,
    write_runtime_selection_benchmark,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class RuntimeSelectionBenchmarkArgs:
    """Validated CLI arguments for the runtime-selection artifact path."""

    config: Path
    output_dir: Path
    run_name: str | None
    v8_artifact_dir: Path | None


def main(argv: Sequence[str] | None = None) -> int:
    """Write local selected-runtime benchmark proof artifacts.

    The local CLI records v8 provenance and writes this benchmark's own failed
    proof when real dual-T4 timing evidence is not supplied by an executor.

    Returns:
        Process exit code.

    """
    args = _parse_args(argv)
    write_runtime_selection_benchmark(
        RuntimeSelectionBenchmarkRequest(
            config_path=args.config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            v8_artifact_dir=args.v8_artifact_dir,
        ),
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> RuntimeSelectionBenchmarkArgs:
    parser = argparse.ArgumentParser(
        description="Write spec 0001 selected-runtime benchmark proof artifacts.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--v8-artifact-dir")
    namespace = parser.parse_args(argv)
    v8_artifact_dir = _optional_str(namespace, "v8_artifact_dir")
    return RuntimeSelectionBenchmarkArgs(
        config=Path(_required_str(namespace, "config")),
        output_dir=Path(_required_str(namespace, "output_dir")),
        run_name=_optional_str(namespace, "run_name"),
        v8_artifact_dir=None if v8_artifact_dir is None else Path(v8_artifact_dir),
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
