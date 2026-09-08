# Copyright 2026 HiperMaximus
# ruff: noqa: DOC201, DOC501, EM101, TRY003
"""Minimal paired initialization and nested-order mechanics for Spec 0023."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

MINIMUM_PAIRED_STEPS = 2

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence


def make_paired_models[ModelT: nn.Module](
    factory: Callable[[], ModelT],
    *,
    seed: int,
) -> tuple[ModelT, ModelT]:
    """Create independent models from one deterministic CPU initialization."""
    with torch.random.fork_rng(devices=[]):  # pyright: ignore[reportUnknownMemberType]
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        reference = factory().cpu()
    return copy.deepcopy(reference), copy.deepcopy(reference)


def nested_tissue_epoch_order(
    full_rows: Sequence[int],
    subset_rows: Collection[int],
    *,
    epoch: int,
    seed: int = 3407,
) -> tuple[int, ...]:
    """Filter one full-pool epoch permutation to a fixed nested subset."""
    if epoch < 0 or len(set(full_rows)) != len(full_rows):
        raise ValueError("Epoch must be nonnegative and full rows unique")
    subset = set(subset_rows)
    if not subset <= set(full_rows):
        raise ValueError("Tissue subset contains a row outside the full pool")
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, epoch])),
    )
    positions = cast("list[int]", generator.permutation(len(full_rows)).tolist())
    return tuple(
        full_rows[position] for position in positions if full_rows[position] in subset
    )


def paired_epoch_order(*, row_count: int, epoch: int, seed: int) -> tuple[int, ...]:
    """Derive one shared without-replacement order for a paired epoch."""
    if row_count < 1 or epoch < 0:
        raise ValueError("Paired epoch geometry must be positive")
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, epoch])),
    )
    return tuple(cast("list[int]", generator.permutation(row_count).tolist()))


def exponential_learning_rates(
    *,
    start: float,
    stop: float,
    update_count: int,
) -> tuple[float, ...]:
    """Return the exact inclusive one-pass LR sweep used by both branches."""
    if not 0.0 < start < stop or update_count < MINIMUM_PAIRED_STEPS:
        raise ValueError("An exponential sweep requires two positive endpoints")
    log_ratio = math.log(stop / start)
    return tuple(
        start * math.exp(log_ratio * index / (update_count - 1))
        for index in range(update_count)
    )


def warmup_update_count(steps_per_epoch: int) -> int:
    """Round the agreed first-10%-of-epoch warmup upward to whole updates."""
    if steps_per_epoch < 1:
        raise ValueError("An epoch must contain a successful optimizer update")
    return math.ceil(steps_per_epoch / 10)


def confirmation_learning_rate(
    *,
    peak: float,
    successful_update: int,
    steps_per_epoch: int,
) -> float:
    """Warm from zero over 10% of epoch one, then hold the sealed peak."""
    if peak <= 0.0 or successful_update < 1:
        raise ValueError("Confirmation LR requires a positive peak and update")
    warmup = warmup_update_count(steps_per_epoch)
    return peak * min(successful_update, warmup) / warmup


def full_training_learning_rate(
    *,
    peak: float,
    successful_update: int,
    steps_per_epoch: int,
    maximum_epochs: int = 30,
    minimum_ratio: float = 0.01,
) -> float:
    """Apply the locked 10% warmup then no-restart cosine decay schedule."""
    total_updates = steps_per_epoch * maximum_epochs
    if (
        peak <= 0.0
        or successful_update < 1
        or successful_update > total_updates
        or maximum_epochs < 1
        or not 0.0 < minimum_ratio < 1.0
    ):
        raise ValueError("Full supervised LR schedule geometry is invalid")
    warmup = warmup_update_count(steps_per_epoch)
    if total_updates <= warmup:
        raise ValueError("Full supervised schedule must extend beyond warmup")
    if successful_update <= warmup:
        return peak * successful_update / warmup
    progress = (successful_update - warmup) / (total_updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def half_epoch_boundaries(steps_per_epoch: int) -> tuple[int, int]:
    """Return the externally named midpoint and endpoint update cursors."""
    if steps_per_epoch < MINIMUM_PAIRED_STEPS:
        raise ValueError("A paired epoch requires distinct half and end boundaries")
    return steps_per_epoch // 2, steps_per_epoch


__all__ = [
    "confirmation_learning_rate",
    "exponential_learning_rates",
    "full_training_learning_rate",
    "half_epoch_boundaries",
    "make_paired_models",
    "nested_tissue_epoch_order",
    "paired_epoch_order",
    "warmup_update_count",
]
