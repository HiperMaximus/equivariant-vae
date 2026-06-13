# Copyright 2026 HiperMaximus
"""Training helpers for spec 0001 local slices."""

from __future__ import annotations

from eqvae.training.optim import (
    OptimizerGroupSummary,
    SpecAdamWConfig,
    build_adamw_parameter_groups,
    create_adamw_optimizer,
)
from eqvae.training.step import TrainStepRequest, TrainStepResult, run_train_step

__all__ = [
    "OptimizerGroupSummary",
    "SpecAdamWConfig",
    "TrainStepRequest",
    "TrainStepResult",
    "build_adamw_parameter_groups",
    "create_adamw_optimizer",
    "run_train_step",
]
