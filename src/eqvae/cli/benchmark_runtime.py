# Copyright 2026 HiperMaximus
"""CLI for the local synthetic runtime-benchmark schema smoke."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eqvae.benchmarking.dataloader_pretest import (
    LocalDataloaderPretestRequest,
    write_local_dataloader_pretest,
)
from eqvae.benchmarking.model_loss_train_step import (
    LocalModelLossTrainStepRequest,
    write_local_model_loss_train_step,
)
from eqvae.benchmarking.runtime_schema import (
    SyntheticBenchmarkRequest,
    write_synthetic_benchmark_artifacts,
)
from eqvae.benchmarking.stain_corruptor_qa import (
    LocalStainCorruptorQaRequest,
    write_local_stain_corruptor_qa,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class BenchmarkRuntimeArgs:
    """Validated arguments for the benchmark-runtime scaffold CLI."""

    config: Path
    data: str
    device: str
    output_dir: Path
    run_name: str
    max_benchmark_rows: int
    warmup_steps: int
    measured_steps: int
    dataloader_pretest: bool
    model_loss_train_step: bool
    stain_corruptor_qa: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local synthetic runtime-benchmark schema smoke.

    Returns:
        Process exit status.

    Raises:
        ValueError: If mutually exclusive local benchmark modes are combined.

    """
    args = _parse_args(argv)
    local_dedicated_modes = (
        args.dataloader_pretest,
        args.model_loss_train_step,
        args.stain_corruptor_qa,
    )
    if sum(1 for enabled in local_dedicated_modes if enabled) > 1:
        message = (
            "`--dataloader-pretest`, `--model-loss-train-step`, and "
            "`--stain-corruptor-qa` are mutually exclusive local modes."
        )
        raise ValueError(message)
    if args.model_loss_train_step:
        write_local_model_loss_train_step(
            LocalModelLossTrainStepRequest(
                config_path=args.config,
                output_dir=args.output_dir,
                run_name=args.run_name,
            ),
        )
        return 0
    if args.stain_corruptor_qa:
        write_local_stain_corruptor_qa(
            LocalStainCorruptorQaRequest(
                config_path=args.config,
                output_dir=args.output_dir,
                run_name=args.run_name,
            ),
        )
        return 0

    write_synthetic_benchmark_artifacts(
        SyntheticBenchmarkRequest(
            config_path=args.config,
            output_dir=args.output_dir,
            run_name=args.run_name,
            max_benchmark_rows=args.max_benchmark_rows,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        ),
    )
    if args.dataloader_pretest:
        write_local_dataloader_pretest(
            LocalDataloaderPretestRequest(
                config_path=args.config,
                output_dir=args.output_dir,
                run_name=args.run_name,
            ),
        )
    return 0


def _parse_args(argv: Sequence[str] | None) -> BenchmarkRuntimeArgs:
    parser = argparse.ArgumentParser(
        description="Write spec 0001 local synthetic benchmark schemas.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", choices=("synthetic",), required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-benchmark-rows", default=2, type=int)
    parser.add_argument("--warmup-steps", default=1, type=int)
    parser.add_argument("--measured-steps", default=2, type=int)
    parser.add_argument(
        "--dataloader-pretest",
        action="store_true",
        help="Measure the local CPU synthetic dataloader pre-test matrix.",
    )
    parser.add_argument(
        "--model-loss-train-step",
        action="store_true",
        help="Run the local CPU synthetic model/loss train-step pre-test.",
    )
    parser.add_argument(
        "--stain-corruptor-qa",
        action="store_true",
        help="Write the local CPU synthetic HED stain-corruptor QA artifact.",
    )
    namespace = parser.parse_args(argv)
    return BenchmarkRuntimeArgs(
        config=Path(_required_str(namespace, "config")),
        data=_required_str(namespace, "data"),
        device=_required_str(namespace, "device"),
        output_dir=Path(_required_str(namespace, "output_dir")),
        run_name=_required_str(namespace, "run_name"),
        max_benchmark_rows=_required_int(namespace, "max_benchmark_rows"),
        warmup_steps=_required_int(namespace, "warmup_steps"),
        measured_steps=_required_int(namespace, "measured_steps"),
        dataloader_pretest=_required_bool(namespace, "dataloader_pretest"),
        model_loss_train_step=_required_bool(namespace, "model_loss_train_step"),
        stain_corruptor_qa=_required_bool(namespace, "stain_corruptor_qa"),
    )


def _required_str(namespace: argparse.Namespace, name: str) -> str:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, str):
        return value
    message = f"Expected string argument: {name}"
    raise TypeError(message)


def _required_int(namespace: argparse.Namespace, name: str) -> int:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, int):
        return value
    message = f"Expected integer argument: {name}"
    raise TypeError(message)


def _required_bool(namespace: argparse.Namespace, name: str) -> bool:
    value = cast("object", getattr(namespace, name))
    if isinstance(value, bool):
        return value
    message = f"Expected boolean argument: {name}"
    raise TypeError(message)


if __name__ == "__main__":
    raise SystemExit(main())
