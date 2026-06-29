# Copyright 2026 HiperMaximus
"""CLI for the selected-runtime debug/resume/tiny gate contract."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.selected_runtime_gate import (
    LOCAL_SELECTOR_MODE,
    REMOTE_GENERATE_MODE,
    SelectedRuntimeGateRequest,
    SelectorGenerationMode,
    verify_selected_runtime_debug_output,
    verify_selected_runtime_debug_push_ready,
    verify_selected_runtime_full_output,
    write_selected_runtime_gate,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class SelectedRuntimeGateArgs:
    """Validated CLI arguments for the selected-runtime gate."""

    debug_config: Path | None
    tiny_config: Path | None
    runtime_config: Path
    output_dir: Path | None
    run_name: str | None
    data_root: str | None
    fixed_train_patches: Path | None
    selector_generation_mode: SelectorGenerationMode
    verify_push_ready: bool
    verify_output: bool
    verify_full_output: bool


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Write fail-closed selected-runtime gate artifacts.

    Returns:
        Process exit status.

    Raises:
        ValueError: If artifact output arguments are missing in write mode.

    """
    args = _parse_args(argv)
    verify_modes = sum(
        int(enabled)
        for enabled in (
            args.verify_push_ready,
            args.verify_output,
            args.verify_full_output,
        )
    )
    if verify_modes > 1:
        message = "verify modes are mutually exclusive"
        raise ValueError(message)
    if args.verify_push_ready:
        debug_config = _required_path_arg(args.debug_config, "--debug-config")
        tiny_config = _required_path_arg(args.tiny_config, "--tiny-config")
        blockers = verify_selected_runtime_debug_push_ready(
            debug_config_path=debug_config,
            tiny_config_path=tiny_config,
            selected_runtime_path=args.runtime_config,
            selector_generation_mode=args.selector_generation_mode,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
        )
        for blocker in blockers:
            sys.stderr.write(f"error: {blocker}\n")
        return 1 if blockers else 0
    if args.verify_output:
        if args.output_dir is None:
            message = "--output-dir is required when --verify-output is set"
            raise ValueError(message)
        blockers = verify_selected_runtime_debug_output(
            output_dir=args.output_dir,
            selected_runtime_path=args.runtime_config,
        )
        for blocker in blockers:
            sys.stderr.write(f"error: {blocker}\n")
        return 1 if blockers else 0
    if args.verify_full_output:
        if args.output_dir is None:
            message = "--output-dir is required when --verify-full-output is set"
            raise ValueError(message)
        blockers = verify_selected_runtime_full_output(
            output_dir=args.output_dir,
            selected_runtime_path=args.runtime_config,
        )
        for blocker in blockers:
            sys.stderr.write(f"error: {blocker}\n")
        return 1 if blockers else 0

    if args.output_dir is None:
        message = "--output-dir is required unless a verify mode is set"
        raise ValueError(message)
    if args.run_name is None:
        message = "--run-name is required unless --verify-push-ready is set"
        raise ValueError(message)
    debug_config = _required_path_arg(args.debug_config, "--debug-config")
    tiny_config = _required_path_arg(args.tiny_config, "--tiny-config")
    write_selected_runtime_gate(
        SelectedRuntimeGateRequest(
            debug_config_path=debug_config,
            tiny_config_path=tiny_config,
            selected_runtime_path=args.runtime_config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            data_root=args.data_root,
            fixed_train_patches=args.fixed_train_patches,
            selector_generation_mode=args.selector_generation_mode,
        ),
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> SelectedRuntimeGateArgs:
    parser = argparse.ArgumentParser(
        description="Write selected-runtime debug/resume/tiny gate artifacts.",
    )
    parser.add_argument("--debug-config")
    parser.add_argument("--tiny-config")
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--data-root")
    parser.add_argument("--fixed-train-patches")
    parser.add_argument(
        "--selector-generation-mode",
        choices=(LOCAL_SELECTOR_MODE, REMOTE_GENERATE_MODE),
        default=LOCAL_SELECTOR_MODE,
    )
    parser.add_argument("--verify-push-ready", action="store_true")
    parser.add_argument("--verify-output", action="store_true")
    parser.add_argument("--verify-full-output", action="store_true")
    namespace = parser.parse_args(argv)
    return SelectedRuntimeGateArgs(
        debug_config=_optional_path(namespace, "debug_config"),
        tiny_config=_optional_path(namespace, "tiny_config"),
        runtime_config=Path(_required_str(namespace, "runtime_config")),
        output_dir=_optional_path(namespace, "output_dir"),
        run_name=_optional_str(namespace, "run_name"),
        data_root=_optional_str(namespace, "data_root"),
        fixed_train_patches=_optional_path(namespace, "fixed_train_patches"),
        selector_generation_mode=cast(
            "SelectorGenerationMode",
            _required_str(namespace, "selector_generation_mode"),
        ),
        verify_push_ready=_required_bool(namespace, "verify_push_ready"),
        verify_output=_required_bool(namespace, "verify_output"),
        verify_full_output=_required_bool(namespace, "verify_full_output"),
    )


def _required_str(namespace: argparse.Namespace, name: str) -> str:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, str):
        return value
    message = f"Expected string argument: {name}"
    raise TypeError(message)


def _required_path_arg(value: Path | None, option: str) -> Path:
    if value is not None:
        return value
    message = f"{option} is required unless --verify-output is set"
    raise ValueError(message)


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
