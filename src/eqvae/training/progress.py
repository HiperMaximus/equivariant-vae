# Copyright 2026 HiperMaximus
"""Progress accounting for optimizer-update-gated training schedules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProgressState:
    """Counters whose schedules advance only after successful optimizer updates."""

    batch_attempt_count: int = 0
    successful_optimizer_update_count: int = 0
    lr_scheduler_step_count: int = 0
    checkpoint_event_count: int = 0
    validation_event_count: int = 0
    tiny_smoothing_update_count: int = 0

    @property
    def optimizer_step_index(self) -> int:
        """The zero-based step index for the next successful update."""
        return self.successful_optimizer_update_count


@dataclass(frozen=True)
class TrainingAttemptProgress:
    """Progress result from one batch attempt."""

    before: TrainingProgressState
    after: TrainingProgressState
    amp_step_skipped: bool
    checkpoint_due: bool
    validation_due: bool
    tiny_smoothing_advanced: bool


def record_training_attempt(
    progress: TrainingProgressState,
    *,
    amp_step_skipped: bool,
    checkpoint_interval: int,
    validation_interval: int,
    tiny_smoothing_enabled: bool,
) -> TrainingAttemptProgress:
    """Record one batch attempt without advancing schedules on AMP skips.

    Returns:
        Progress before/after plus event flags.

    Raises:
        ValueError: If an interval is negative.

    """
    if checkpoint_interval < 0:
        message = f"checkpoint_interval must be nonnegative, got {checkpoint_interval}"
        raise ValueError(message)
    if validation_interval < 0:
        message = f"validation_interval must be nonnegative, got {validation_interval}"
        raise ValueError(message)

    attempted = TrainingProgressState(
        batch_attempt_count=progress.batch_attempt_count + 1,
        successful_optimizer_update_count=progress.successful_optimizer_update_count,
        lr_scheduler_step_count=progress.lr_scheduler_step_count,
        checkpoint_event_count=progress.checkpoint_event_count,
        validation_event_count=progress.validation_event_count,
        tiny_smoothing_update_count=progress.tiny_smoothing_update_count,
    )
    if amp_step_skipped:
        return TrainingAttemptProgress(
            before=progress,
            after=attempted,
            amp_step_skipped=True,
            checkpoint_due=False,
            validation_due=False,
            tiny_smoothing_advanced=False,
        )

    successful_count = progress.successful_optimizer_update_count + 1
    checkpoint_due = (
        checkpoint_interval > 0 and successful_count % checkpoint_interval == 0
    )
    validation_due = (
        validation_interval > 0 and successful_count % validation_interval == 0
    )
    tiny_smoothing_advanced = tiny_smoothing_enabled
    after = TrainingProgressState(
        batch_attempt_count=attempted.batch_attempt_count,
        successful_optimizer_update_count=successful_count,
        lr_scheduler_step_count=progress.lr_scheduler_step_count + 1,
        checkpoint_event_count=(
            progress.checkpoint_event_count + (1 if checkpoint_due else 0)
        ),
        validation_event_count=(
            progress.validation_event_count + (1 if validation_due else 0)
        ),
        tiny_smoothing_update_count=(
            progress.tiny_smoothing_update_count + (1 if tiny_smoothing_advanced else 0)
        ),
    )
    return TrainingAttemptProgress(
        before=progress,
        after=after,
        amp_step_skipped=False,
        checkpoint_due=checkpoint_due,
        validation_due=validation_due,
        tiny_smoothing_advanced=tiny_smoothing_advanced,
    )


__all__ = [
    "TrainingAttemptProgress",
    "TrainingProgressState",
    "record_training_attempt",
]
