# Copyright 2026 HiperMaximus
"""CLI for the selected-runtime debug/resume/tiny gate contract."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.selected_runtime_gate import (
    SelectedRuntimeGateRequest,
    verify_selected_runtime_debug_push_ready,
    write_selected_runtime_gate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SelectedRuntimeGateArgs:
    """Validated CLI arguments for the selected-runtime gate."""

    debug_config: Path
    tiny_config: Path
    runtime_config: Path
    output_dir: Path | None
    run_name: str | None
    data_root: str | None
    fixed_train_patches: Path | None
    verify_push_ready: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Write fail-closed selected-runtime gate artifacts.

    Returns:
        Process exit status.

    Raises:
        ValueError: If artifact output arguments are missing in write mode.

    """
    args = _parse_args(argv)
    if args.verify_push_ready:
        blockers = verify_selected_runtime_debug_push_ready(
            debug_config_path=args.debug_config,
            tiny_config_path=args.tiny_config,
            selected_runtime_path=args.runtime_config,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
        )
        for blocker in blockers:
            sys.stderr.write(f"error: {blocker}\n")
        return 1 if blockers else 0

    if args.output_dir is None:
        message = "--output-dir is required unless --verify-push-ready is set"
        raise ValueError(message)
    if args.run_name is None:
        message = "--run-name is required unless --verify-push-ready is set"
        raise ValueError(message)
    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=args.debug_config,
            tiny_config_path=args.tiny_config,
            selected_runtime_path=args.runtime_config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
        ),
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> SelectedRuntimeGateArgs:
    parser = argparse.ArgumentParser(
        description="Write selected-runtime debug/resume/tiny gate artifacts.",
    )
    parser.add_argument("--debug-config", required=True)
    parser.add_argument("--tiny-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--data-root")
    parser.add_argument("--fixed-train-patches")
    parser.add_argument("--verify-push-ready", action="store_true")
    namespace = parser.parse_args(argv)
    return SelectedRuntimeGateArgs(
        debug_config=Path(_required_str(namespace, "debug_config")),
        tiny_config=Path(_required_str(namespace, "tiny_config")),
        runtime_config=Path(_required_str(namespace, "runtime_config")),
        output_dir=_optional_path(namespace, "output_dir"),
        run_name=_optional_str(namespace, "run_name"),
        data_root=_optional_str(namespace, "data_root"),
        fixed_train_patches=_optional_path(namespace, "fixed_train_patches"),
        verify_push_ready=_required_bool(namespace, "verify_push_ready"),
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


def _optional_path(namespace: argparse.Namespace, name: str) -> Path | None:
    value = _optional_str(namespace, name)
    return None if value is None else Path(value)


def _required_bool(namespace: argparse.Namespace, name: str) -> bool:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, bool):
        return value
    message = f"Expected boolean argument: {name}"
    raise TypeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
