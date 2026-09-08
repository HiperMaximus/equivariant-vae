# Copyright 2026 HiperMaximus
# ruff: noqa: BLE001, C901, DOC201, DOC501, EM101, EM102, PLR0912, PLR0913, PLR0914, PLR0915, PLR0916, PLR2004, PLW0717, S311, TRY003, TRY004, TRY301
"""Branch-local training, validation, and checkpoint primitives for Spec 0036."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn

from eqvae.models.local_global_mil import (
    CLASS_ORDER,
    LocalGlobalMILClassifier,
    local_global_mil_adamw_parameter_groups,
)
from eqvae.training.supervised_pairing import (
    half_epoch_boundaries,
    paired_epoch_order,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch.amp.grad_scaler import GradScaler

TRAINING_SEED: Final = 1701
BOOTSTRAP_SEED: Final = 3601
TRAIN_WSI_COUNT: Final = 106
VALIDATION_WSI_COUNT: Final = 23
MAXIMUM_EPOCHS: Final = 150
WARMUP_EPOCHS: Final = 5
WARMUP_UPDATES: Final = TRAIN_WSI_COUNT * WARMUP_EPOCHS
WARMUP_START_RATIO: Final = 0.1
TOTAL_UPDATES: Final = TRAIN_WSI_COUNT * MAXIMUM_EPOCHS
PEAK_LEARNING_RATE: Final = 2e-4
WARMUP_START_LEARNING_RATE: Final = PEAK_LEARNING_RATE * WARMUP_START_RATIO
MINIMUM_LEARNING_RATE_RATIO: Final = 0.01
WEIGHT_DECAY: Final = 5e-3
ADAMW_BETAS: Final = (0.9, 0.999)
ADAMW_EPSILON: Final = 1e-8
EARLY_STOPPING_START_EPOCH: Final = 50
EARLY_STOPPING_START_UPDATE: Final = TRAIN_WSI_COUNT * EARLY_STOPPING_START_EPOCH
PATIENCE_EPOCHS: Final = 20
PATIENCE_CHECKS: Final = 2 * PATIENCE_EPOCHS
MAX_SCALE_BACKOFFS: Final = 3
SCALER_INITIAL_SCALE: Final = 65_536.0
SCALER_GROWTH_FACTOR: Final = 2.0
SCALER_BACKOFF_FACTOR: Final = 0.5
SCALER_GROWTH_INTERVAL: Final = 2_000
BOOTSTRAP_REPLICATES: Final = 10_000
CHECKPOINT_SCHEMA: Final = "spec0036.branch_checkpoint.v1"
CHECKPOINT_MANIFEST_SCHEMA: Final = "spec0036.branch_manifest.v1"
CHECKPOINT_POINTER_SCHEMA: Final = "spec0036.branch_pointer.v1"
CHECKPOINT_SLOTS: Final = ("latest", "best", "final")
BRANCH_NAMES: Final = ("normal_vae", "so2_vae")
TRAIN_CLASS_COUNTS: Final = (23, 24, 40, 11, 8)
CLASS_WEIGHTS: Final = tuple(
    TRAIN_WSI_COUNT / (len(CLASS_ORDER) * count) for count in TRAIN_CLASS_COUNTS
)
CLASS_WEIGHT_BY_INDEX: Final = dict(enumerate(CLASS_WEIGHTS))
CLASS_WEIGHT_BY_DIAGNOSIS: Final = dict(zip(CLASS_ORDER, CLASS_WEIGHTS, strict=True))
CALIBRATION_WSI_IDS: Final = (45_630, 51_346, 57_162, 65_094, 35_239)
CALIBRATION_DIAGNOSES: Final = ("EC", "CC", "LGSC", "MC", "HGSC")


@dataclass(frozen=True, init=False)
class MILTrainingConfig:
    """Expose the locked Spec 0036 training policy without tuning knobs."""

    construction_seed: int = TRAINING_SEED
    bootstrap_seed: int = BOOTSTRAP_SEED
    train_wsi_count: int = TRAIN_WSI_COUNT
    validation_wsi_count: int = VALIDATION_WSI_COUNT
    maximum_epochs: int = MAXIMUM_EPOCHS
    peak_learning_rate: float = PEAK_LEARNING_RATE
    minimum_learning_rate_ratio: float = MINIMUM_LEARNING_RATE_RATIO
    warmup_epochs: int = WARMUP_EPOCHS
    warmup_updates: int = WARMUP_UPDATES
    warmup_start_ratio: float = WARMUP_START_RATIO
    warmup_start_learning_rate: float = WARMUP_START_LEARNING_RATE
    total_updates: int = TOTAL_UPDATES
    weight_decay: float = WEIGHT_DECAY
    adamw_betas: tuple[float, float] = ADAMW_BETAS
    adamw_epsilon: float = ADAMW_EPSILON
    early_stopping_start_epoch: int = EARLY_STOPPING_START_EPOCH
    early_stopping_start_update: int = EARLY_STOPPING_START_UPDATE
    patience_epochs: int = PATIENCE_EPOCHS
    patience_checks: int = PATIENCE_CHECKS
    maximum_scale_backoffs: int = MAX_SCALE_BACKOFFS
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    class_order: tuple[str, ...] = CLASS_ORDER
    train_class_counts: tuple[int, ...] = TRAIN_CLASS_COUNTS
    class_weights: tuple[float, ...] = CLASS_WEIGHTS
    calibration_wsi_ids: tuple[int, ...] = CALIBRATION_WSI_IDS


MIL_TRAINING_CONFIG: Final = MILTrainingConfig()


@dataclass(frozen=True)
class ClassMetrics:
    """Validation measures for one diagnosis in canonical class order."""

    diagnosis: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ValidationMetrics:
    """Complete WSI-level validation summary used by selection and reporting."""

    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    mean_ce: float
    per_class: tuple[ClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    wsi_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe metric record with explicit class identities."""
        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True)
class BestSelectionState:
    """Branch-owned selector and early-stopping state."""

    best_metrics: ValidationMetrics | None = None
    best_boundary: int | None = None
    non_improving_checks: int = 0
    stopped: bool = False


@dataclass(frozen=True)
class SelectionUpdate:
    """Result of applying one validation boundary to one branch selector."""

    state: BestSelectionState
    improved: bool


@dataclass(frozen=True)
class BootstrapInterval:
    """Observed normal-minus-SO(2) difference and percentile interval."""

    difference: float
    confidence_low: float
    confidence_high: float
    replicates: int


@dataclass(frozen=True)
class RNGState:
    """Exact process RNG state needed by calibration and branch resume."""

    python: object
    numpy: object
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...]


@dataclass(frozen=True)
class WeightedAMPStepResult:
    """Observation emitted only after one optimizer update commits."""

    learning_rate: float
    unweighted_loss: float
    weighted_loss: float
    attempts: int
    scale_backoffs: int
    initial_scale: float
    final_scale: float


@dataclass(frozen=True)
class CalibrationCaseResult:
    """One committed stress-WSI result and the scaler state it produced."""

    wsi_id: int
    step: WeightedAMPStepResult
    scaler_state: dict[str, object]


@dataclass(frozen=True)
class CalibrationResult:
    """Auditable calibration evidence whose only retained runtime state is scaler."""

    scaler: StepScaler
    cases: tuple[CalibrationCaseResult, ...]


@dataclass(frozen=True)
class LoadedBranchCheckpoint:
    """A validated checkpoint payload and its immutable object identity."""

    payload: dict[str, object]
    checkpoint_sha256: str
    manifest_sha256: str
    slot: str


class BranchNumericalError(RuntimeError):
    """Terminal branch-local numerical failure with named diagnostics."""

    def __init__(self, message: str, *, details: Mapping[str, object]) -> None:
        """Retain compact structured evidence for the terminal branch record."""
        super().__init__(message)
        self.details = dict(details)


class StepScaler(Protocol):
    """Small GradScaler surface used by the branch step and CPU fakes."""

    def scale(self, output: Tensor) -> Tensor:
        """Scale a scalar loss."""
        raise NotImplementedError

    def step(self, optimizer: torch.optim.Optimizer) -> object | None:
        """Commit or skip an optimizer update."""
        raise NotImplementedError

    def update(self, new_scale: float | Tensor | None = None) -> None:
        """Advance the dynamic scale lifecycle."""
        raise NotImplementedError

    def get_scale(self) -> float:
        """Return the current scale."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        """Return complete scaler state."""
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore complete scaler state."""
        raise NotImplementedError


def committed_update_learning_rate(committed_update: int) -> float:
    """Return the exact locked LR indexed only by a committed update count."""
    if committed_update < 1 or committed_update > TOTAL_UPDATES:
        raise ValueError("Committed update is outside the locked MIL schedule")
    if committed_update <= WARMUP_UPDATES:
        progress = (committed_update - 1) / (WARMUP_UPDATES - 1)
        ratio = WARMUP_START_RATIO + (1.0 - WARMUP_START_RATIO) * progress
        return PEAK_LEARNING_RATE * ratio
    decay_progress = (committed_update - WARMUP_UPDATES) / (
        TOTAL_UPDATES - WARMUP_UPDATES
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    ratio = MINIMUM_LEARNING_RATE_RATIO + (1.0 - MINIMUM_LEARNING_RATE_RATIO) * cosine
    return PEAK_LEARNING_RATE * ratio


def class_weight_for_label(label: int) -> float:
    """Return the exact inverse-frequency scalar for one canonical class index."""
    try:
        return CLASS_WEIGHT_BY_INDEX[label]
    except KeyError as error:
        raise ValueError("MIL label is outside the canonical class order") from error


def training_epoch_order(epoch: int) -> tuple[int, ...]:
    """Return the branch-independent permutation for a zero-based epoch."""
    return paired_epoch_order(
        row_count=TRAIN_WSI_COUNT,
        epoch=epoch,
        seed=TRAINING_SEED,
    )


def is_validation_boundary(committed_update: int) -> bool:
    """Recognize only the midpoint and endpoint of a complete 106-WSI epoch."""
    if committed_update < 1 or committed_update > TOTAL_UPDATES:
        return False
    position = committed_update % TRAIN_WSI_COUNT
    midpoint, endpoint = half_epoch_boundaries(TRAIN_WSI_COUNT)
    return position == midpoint or position == endpoint % TRAIN_WSI_COUNT


def update_best_selection(
    state: BestSelectionState,
    metrics: ValidationMetrics,
    *,
    boundary: int,
) -> SelectionUpdate:
    """Update one branch by macro-F1, CE, then earlier-boundary ordering."""
    if boundary < 1 or not is_validation_boundary(boundary):
        raise ValueError("Selection may change only at a committed validation boundary")
    _validate_finite_metrics(metrics)
    if state.stopped:
        raise ValueError("A stopped branch cannot consume another validation check")
    if state.best_metrics is None:
        improved = True
    else:
        if state.best_boundary is None:
            raise ValueError("Best metrics require their selected boundary")
        improved = _selection_key(metrics, boundary) > _selection_key(
            state.best_metrics,
            state.best_boundary,
        )
    patience_active = boundary > EARLY_STOPPING_START_UPDATE
    non_improving = (
        state.non_improving_checks + 1 if patience_active and not improved else 0
    )
    next_state = BestSelectionState(
        best_metrics=metrics if improved else state.best_metrics,
        best_boundary=boundary if improved else state.best_boundary,
        non_improving_checks=non_improving,
        stopped=non_improving >= PATIENCE_CHECKS,
    )
    return SelectionUpdate(state=next_state, improved=improved)


def compute_validation_metrics(
    truths: Sequence[int],
    predictions: Sequence[int],
    cross_entropies: Sequence[float],
) -> ValidationMetrics:
    """Compute all locked WSI metrics without a scikit-learn dependency."""
    if (
        not truths
        or len(truths) != len(predictions)
        or len(truths) != len(cross_entropies)
    ):
        raise ValueError("Validation truth, prediction, and CE rows must align")
    class_count = len(CLASS_ORDER)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, prediction in zip(truths, predictions, strict=True):
        if not 0 <= truth < class_count or not 0 <= prediction < class_count:
            raise ValueError("Validation labels must use the canonical class indices")
        confusion[truth][prediction] += 1
    losses = np.asarray(cross_entropies, dtype=np.float64)
    if losses.ndim != 1 or not bool(np.isfinite(losses).all()):
        raise ValueError("Validation cross-entropies must be finite scalars")

    per_class: list[ClassMetrics] = []
    for class_index, diagnosis in enumerate(CLASS_ORDER):
        true_positive = confusion[class_index][class_index]
        support = sum(confusion[class_index])
        predicted = sum(row[class_index] for row in confusion)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            ClassMetrics(
                diagnosis=diagnosis,
                precision=precision,
                recall=recall,
                f1=f1,
                support=support,
            ),
        )
    correct = sum(confusion[index][index] for index in range(class_count))
    return ValidationMetrics(
        macro_f1=sum(item.f1 for item in per_class) / class_count,
        balanced_accuracy=sum(item.recall for item in per_class) / class_count,
        accuracy=correct / len(truths),
        mean_ce=float(losses.mean()),
        per_class=tuple(per_class),
        confusion_matrix=tuple(tuple(row) for row in confusion),
        wsi_count=len(truths),
    )


def predictions_from_logits(logits: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Convert finite five-logit WSI records to canonical class predictions."""
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(CLASS_ORDER):
        raise ValueError("Validation logits must have shape [N,5]")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Validation logits must be finite")
    return tuple(cast("list[int]", array.argmax(axis=1).tolist()))


def diagnosis_stratified_paired_bootstrap(
    truths: Sequence[int],
    normal_predictions: Sequence[int],
    so2_predictions: Sequence[int],
    *,
    normal_cross_entropies: Sequence[float] | None = None,
    so2_cross_entropies: Sequence[float] | None = None,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, BootstrapInterval]:
    """Bootstrap paired branch differences within each diagnosis stratum."""
    row_count = len(truths)
    if (
        row_count < 1
        or len(normal_predictions) != row_count
        or len(so2_predictions) != row_count
    ):
        raise ValueError("Paired bootstrap prediction rows must align")
    if replicates < 1:
        raise ValueError("Paired bootstrap requires at least one replicate")
    use_loss = normal_cross_entropies is not None or so2_cross_entropies is not None
    if use_loss and (normal_cross_entropies is None or so2_cross_entropies is None):
        raise ValueError("Paired bootstrap CE rows must be supplied for both branches")
    normal_losses = (
        tuple(float(value) for value in normal_cross_entropies)
        if normal_cross_entropies is not None
        else tuple(0.0 for _ in range(row_count))
    )
    so2_losses = (
        tuple(float(value) for value in so2_cross_entropies)
        if so2_cross_entropies is not None
        else tuple(0.0 for _ in range(row_count))
    )
    if len(normal_losses) != row_count or len(so2_losses) != row_count:
        raise ValueError("Paired bootstrap CE rows must align")
    normal_observed = compute_validation_metrics(
        truths,
        normal_predictions,
        normal_losses,
    )
    so2_observed = compute_validation_metrics(truths, so2_predictions, so2_losses)
    metric_names = ["macro_f1", "balanced_accuracy", "accuracy"]
    if use_loss:
        metric_names.append("mean_ce")
    samples = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    strata = _diagnosis_strata(truths)
    generator = np.random.Generator(np.random.PCG64(seed))
    truths_tuple = tuple(truths)
    normal_predictions_tuple = tuple(normal_predictions)
    so2_predictions_tuple = tuple(so2_predictions)
    for replicate in range(replicates):
        sampled_indices = tuple(
            index
            for stratum in strata
            for index in cast(
                "list[int]",
                generator.choice(stratum, size=len(stratum), replace=True).tolist(),
            )
        )
        sampled_truths = tuple(truths_tuple[index] for index in sampled_indices)
        normal_metrics = compute_validation_metrics(
            sampled_truths,
            tuple(normal_predictions_tuple[index] for index in sampled_indices),
            tuple(normal_losses[index] for index in sampled_indices),
        )
        so2_metrics = compute_validation_metrics(
            sampled_truths,
            tuple(so2_predictions_tuple[index] for index in sampled_indices),
            tuple(so2_losses[index] for index in sampled_indices),
        )
        for name in metric_names:
            samples[name][replicate] = cast(
                "float",
                getattr(normal_metrics, name),
            ) - cast("float", getattr(so2_metrics, name))
    return {
        name: BootstrapInterval(
            difference=cast("float", getattr(normal_observed, name))
            - cast("float", getattr(so2_observed, name)),
            confidence_low=float(np.percentile(values, 2.5)),
            confidence_high=float(np.percentile(values, 97.5)),
            replicates=replicates,
        )
        for name, values in samples.items()
    }


def capture_rng_state(
    *,
    cuda_state_getter: Callable[[], Sequence[Tensor]] | None = None,
) -> RNGState:
    """Capture Python, NumPy, Torch CPU, and available CUDA RNG states."""
    if cuda_state_getter is None:
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )
    else:
        cuda_states = cuda_state_getter()
    return RNGState(
        python=copy.deepcopy(random.getstate()),
        numpy=copy.deepcopy(np.random.get_state()),  # noqa: NPY002
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=tuple(state.detach().cpu().clone() for state in cuda_states),
    )


def restore_rng_state(
    state: RNGState,
    *,
    cuda_state_setter: Callable[[Sequence[Tensor]], None] | None = None,
) -> None:
    """Restore a complete RNG snapshot without silently dropping CUDA state."""
    random.setstate(cast("tuple[Any, ...]", copy.deepcopy(state.python)))
    np.random.set_state(  # noqa: NPY002
        copy.deepcopy(state.numpy),  # pyright: ignore[reportArgumentType]
    )
    torch.set_rng_state(state.torch_cpu.clone())
    if not state.torch_cuda:
        return
    cuda_states = tuple(item.clone() for item in state.torch_cuda)
    if cuda_state_setter is not None:
        cuda_state_setter(cuda_states)
    elif torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(cuda_states))
    else:
        raise ValueError("CUDA RNG state cannot be restored without a CUDA runtime")


def make_default_grad_scaler(*, enabled: bool = True) -> GradScaler:
    """Create the ordinary CUDA GradScaler with its pinned runtime defaults."""
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    validate_default_grad_scaler(scaler, expected_enabled=enabled)
    return scaler


def validate_default_grad_scaler(
    scaler: GradScaler,
    *,
    expected_enabled: bool = True,
) -> None:
    """Fail if a scaler differs from the accepted ordinary PyTorch defaults."""
    if bool(scaler.is_enabled()) != expected_enabled:
        raise ValueError("GradScaler enabled state differs from the branch policy")
    if not expected_enabled:
        return
    observed = (
        float(scaler.get_scale()),
        float(scaler.get_growth_factor()),
        float(scaler.get_backoff_factor()),
        int(scaler.get_growth_interval()),
    )
    expected = (
        SCALER_INITIAL_SCALE,
        SCALER_GROWTH_FACTOR,
        SCALER_BACKOFF_FACTOR,
        SCALER_GROWTH_INTERVAL,
    )
    if observed != expected:
        raise ValueError(
            f"GradScaler defaults differ: expected {expected}, got {observed}",
        )


def make_fused_adamw(
    model: LocalGlobalMILClassifier,
    *,
    learning_rate: float = PEAK_LEARNING_RATE,
    fused: bool = True,
    capturable: bool = False,
) -> torch.optim.AdamW:
    """Create AdamW from the model's semantic decay partition."""
    if learning_rate <= 0.0:
        raise ValueError("AdamW learning rate must be positive")
    groups = local_global_mil_adamw_parameter_groups(
        model,
        weight_decay=WEIGHT_DECAY,
    )
    return torch.optim.AdamW(
        [dict(group) for group in groups],
        lr=learning_rate,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPSILON,
        fused=fused,
        capturable=capturable,
    )


def weighted_amp_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: StepScaler,
    loss_closure: Callable[[], Tensor],
    *,
    class_weight: float,
    learning_rate: float,
    max_scale_backoffs: int = MAX_SCALE_BACKOFFS,
) -> WeightedAMPStepResult:
    """Commit one standard weighted AMP update, retrying the same loaded WSI."""
    if not math.isfinite(class_weight) or class_weight <= 0.0:
        raise ValueError("MIL class weight must be finite and positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("MIL learning rate must be finite and positive")
    if max_scale_backoffs < 0:
        raise ValueError("Maximum scaler backoffs may not be negative")
    initial_scale = float(scaler.get_scale())
    backoffs = 0
    attempts = 0
    while True:
        attempts += 1
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loss = loss_closure()
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
            optimizer.zero_grad(set_to_none=True)
            raise BranchNumericalError(
                "Branch produced a nonfinite scalar loss",
                details={"kind": "scalar_loss", "attempt": attempts},
            )
        weighted_loss = loss.float() * class_weight
        scaler.scale(weighted_loss).backward()  # pyright: ignore[reportUnknownMemberType]
        scale_before_step = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        final_scale = float(scaler.get_scale())
        if final_scale < scale_before_step:
            nonfinite = _named_nonfinite_gradients(model)
            optimizer.zero_grad(set_to_none=True)
            backoffs += 1
            if backoffs >= max_scale_backoffs:
                raise BranchNumericalError(
                    "Branch exhausted same-WSI AMP overflow retries",
                    details={
                        "kind": "gradient_overflow",
                        "named_gradients": nonfinite,
                        "attempts": attempts,
                        "scale_backoffs": backoffs,
                    },
                )
            continue
        unweighted_loss = float(loss.detach().item())
        return WeightedAMPStepResult(
            learning_rate=learning_rate,
            unweighted_loss=unweighted_loss,
            weighted_loss=class_weight * unweighted_loss,
            attempts=attempts,
            scale_backoffs=backoffs,
            initial_scale=initial_scale,
            final_scale=final_scale,
        )


def calibrate_grad_scaler(
    model: nn.Module,
    scaler: StepScaler,
    optimizer_factory: Callable[[], torch.optim.Optimizer],
    loss_for_wsi: Callable[[int], Tensor],
    *,
    class_weights_by_wsi: Mapping[int, float],
    learning_rate: float = PEAK_LEARNING_RATE,
    stress_wsi_ids: Sequence[int] = CALIBRATION_WSI_IDS,
    step: Callable[..., WeightedAMPStepResult] = weighted_amp_step,
) -> CalibrationResult:
    """Calibrate only the scaler across isolated disposable optimizer attempts."""
    if tuple(stress_wsi_ids) != CALIBRATION_WSI_IDS:
        raise ValueError("Calibration WSI order differs from the locked stress order")
    if set(class_weights_by_wsi) != set(CALIBRATION_WSI_IDS):
        raise ValueError("Calibration class weights must cover exactly the stress WSIs")
    initial_model = _clone_model_state(model)
    initial_rng = capture_rng_state()
    cases: list[CalibrationCaseResult] = []
    failure: BaseException | None = None
    try:
        for wsi_id in stress_wsi_ids:
            model.load_state_dict(initial_model)
            model.zero_grad(set_to_none=True)
            restore_rng_state(initial_rng)
            optimizer = optimizer_factory()
            if optimizer.state:
                raise ValueError(
                    "Each calibration optimizer must start with empty state",
                )
            step_result = step(
                model,
                optimizer,
                scaler,
                lambda wsi_id=wsi_id: loss_for_wsi(wsi_id),
                class_weight=class_weights_by_wsi[wsi_id],
                learning_rate=learning_rate,
                max_scale_backoffs=MAX_SCALE_BACKOFFS,
            )
            cases.append(
                CalibrationCaseResult(
                    wsi_id=wsi_id,
                    step=step_result,
                    scaler_state=copy.deepcopy(scaler.state_dict()),
                ),
            )
    except BaseException as error:
        failure = error
    try:
        model.load_state_dict(initial_model)
        model.zero_grad(set_to_none=True)
        restore_rng_state(initial_rng)
        if not _model_state_equal(model, initial_model) or any(
            parameter.grad is not None for parameter in model.parameters()
        ):
            raise RuntimeError("Calibration did not restore a pristine model boundary")
    except BaseException as restoration_error:
        if failure is not None:
            failure.add_note(
                f"Calibration restoration also failed: {restoration_error!r}",
            )
            raise failure from restoration_error
        raise
    if failure is not None:
        raise failure
    return CalibrationResult(scaler=scaler, cases=tuple(cases))


def branch_checkpoint_payload(
    *,
    branch_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: StepScaler,
    committed_update: int,
    epoch: int,
    within_epoch_cursor: int,
    current_order: Sequence[int],
    train_history: Sequence[Mapping[str, object]],
    validation_history: Sequence[Mapping[str, object]],
    best_selection: BestSelectionState,
    contract_hashes: Mapping[str, str],
    access_transcript: Sequence[Mapping[str, object]],
    rng_state: RNGState | None = None,
) -> dict[str, object]:
    """Build the complete branch-owned state required for exact continuation."""
    if branch_name not in BRANCH_NAMES:
        raise ValueError("Checkpoint branch identity is not recognized")
    order = tuple(int(value) for value in current_order)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "branch_name": branch_name,
        "model_state_dict": _clone_model_state(model),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "grad_scaler_state_dict": copy.deepcopy(scaler.state_dict()),
        "committed_update": committed_update,
        "schedule_position": committed_update,
        "epoch": epoch,
        "within_epoch_cursor": within_epoch_cursor,
        "current_order": order,
        "order_sha256": _order_sha256(order),
        "train_history": tuple(dict(row) for row in train_history),
        "validation_history": tuple(dict(row) for row in validation_history),
        "best_selection": _selection_payload(best_selection),
        "patience": best_selection.non_improving_checks,
        "contract_hashes": dict(contract_hashes),
        "access_transcript": tuple(dict(row) for row in access_transcript),
        "rng_state": _rng_payload(rng_state or capture_rng_state()),
    }
    validate_branch_checkpoint_payload(payload)
    return payload


def validate_branch_checkpoint_payload(
    payload: Mapping[str, object],
    *,
    expected_branch_name: str | None = None,
    expected_contract_hashes: Mapping[str, str] | None = None,
) -> None:
    """Reject partial, foreign, stale, or internally inconsistent branch state."""
    required = {
        "schema_version",
        "branch_name",
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
        "committed_update",
        "schedule_position",
        "epoch",
        "within_epoch_cursor",
        "current_order",
        "order_sha256",
        "train_history",
        "validation_history",
        "best_selection",
        "patience",
        "contract_hashes",
        "access_transcript",
        "rng_state",
    }
    if set(payload) != required:
        raise ValueError("Branch checkpoint fields are incomplete or unexpected")
    if payload["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("Unexpected branch checkpoint schema")
    branch_name = payload["branch_name"]
    if branch_name not in BRANCH_NAMES:
        raise ValueError("Branch checkpoint identity is not recognized")
    if expected_branch_name is not None and branch_name != expected_branch_name:
        raise ValueError("Cross-branch checkpoint resume is forbidden")
    committed_update = payload["committed_update"]
    schedule_position = payload["schedule_position"]
    epoch = payload["epoch"]
    cursor = payload["within_epoch_cursor"]
    if (
        not isinstance(committed_update, int)
        or not 0 <= committed_update <= TOTAL_UPDATES
        or schedule_position != committed_update
        or not isinstance(epoch, int)
        or not 0 <= epoch <= MAXIMUM_EPOCHS
        or not isinstance(cursor, int)
        or not 0 <= cursor <= TRAIN_WSI_COUNT
    ):
        raise ValueError("Branch checkpoint progress is invalid")
    if committed_update and not is_validation_boundary(committed_update):
        raise ValueError(
            "A resumable checkpoint must be a committed half-epoch boundary",
        )
    expected_epoch, expected_cursor = divmod(committed_update, TRAIN_WSI_COUNT)
    if epoch != expected_epoch or cursor != expected_cursor:
        raise ValueError(
            "Branch checkpoint epoch/cursor differs from committed progress",
        )
    order_value = payload["current_order"]
    if not isinstance(order_value, (tuple, list)):
        raise TypeError("Branch checkpoint order must be a sequence")
    order = cast("Sequence[object]", order_value)
    if _order_sha256(order) != payload["order_sha256"]:
        raise ValueError("Branch checkpoint order identity differs")
    if len(order) != TRAIN_WSI_COUNT or set(order) != set(range(TRAIN_WSI_COUNT)):
        raise ValueError("Branch checkpoint order is not a complete epoch permutation")
    if tuple(order) != training_epoch_order(epoch):
        raise ValueError("Branch checkpoint order differs from its saved epoch")
    train_history_value = payload["train_history"]
    validation_history_value = payload["validation_history"]
    if not isinstance(train_history_value, (tuple, list)) or not isinstance(
        validation_history_value,
        (tuple, list),
    ):
        raise TypeError("Branch checkpoint histories must be sequences")
    train_history = cast("Sequence[object]", train_history_value)
    validation_history = cast("Sequence[object]", validation_history_value)
    if len(train_history) != committed_update:
        raise ValueError(
            "Branch checkpoint training history differs from committed progress",
        )
    for expected_update, row_value in enumerate(train_history, start=1):
        if not isinstance(row_value, dict):
            raise ValueError("Branch checkpoint training history is not contiguous")
        train_row = cast("Mapping[str, object]", row_value)
        if train_row.get("committed_update") != expected_update:
            raise ValueError("Branch checkpoint training history is not contiguous")
    half_epoch = TRAIN_WSI_COUNT // 2
    expected_boundaries = tuple(range(half_epoch, committed_update + 1, half_epoch))
    observed_boundaries: list[int] = []
    for row_value in validation_history:
        if not isinstance(row_value, dict):
            raise ValueError("Branch checkpoint validation history is malformed")
        validation_row = cast("Mapping[str, object]", row_value)
        boundary_value = validation_row.get("boundary")
        if not isinstance(boundary_value, int):
            raise ValueError("Branch checkpoint validation history is malformed")
        observed_boundaries.append(boundary_value)
    if tuple(observed_boundaries) != expected_boundaries:
        raise ValueError(
            "Branch checkpoint validation history differs from committed progress",
        )
    selection_value = payload["best_selection"]
    if not isinstance(selection_value, dict):
        raise ValueError("Branch checkpoint selector state is invalid")
    selection = _selection_from_payload(cast("Mapping[str, object]", selection_value))
    if payload["patience"] != selection.non_improving_checks:
        raise ValueError("Branch checkpoint selector state is invalid")
    if committed_update == 0:
        if selection != BestSelectionState():
            raise ValueError("An unstarted checkpoint cannot have selector state")
    else:
        recomputed = BestSelectionState()
        for expected_boundary, row_value in zip(
            observed_boundaries,
            validation_history,
            strict=True,
        ):
            row = cast("Mapping[str, object]", row_value)
            metric_payload = {
                key: row.get(key) for key in ValidationMetrics.__dataclass_fields__
            }
            metrics = _validation_metrics_from_payload(metric_payload)
            update = update_best_selection(
                recomputed,
                metrics,
                boundary=expected_boundary,
            )
            improved = row.get("improved")
            non_improving = row.get("non_improving_checks")
            if (
                not isinstance(improved, bool)
                or improved != update.improved
                or not isinstance(non_improving, int)
                or isinstance(non_improving, bool)
                or non_improving != update.state.non_improving_checks
            ):
                raise ValueError(
                    "Branch checkpoint validation selector history differs",
                )
            expected_validation_epoch = (expected_boundary - 1) // TRAIN_WSI_COUNT
            expected_validation_cursor = (expected_boundary - 1) % TRAIN_WSI_COUNT + 1
            if (
                row.get("epoch") != expected_validation_epoch
                or row.get("within_epoch_cursor") != expected_validation_cursor
            ):
                raise ValueError(
                    "Branch checkpoint validation progress history differs",
                )
            recomputed = update.state
        if recomputed != selection:
            raise ValueError(
                "Branch checkpoint best selection differs from validation history",
            )
    hashes = payload["contract_hashes"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Branch checkpoint contract hashes are missing")
    if expected_contract_hashes is not None and hashes != dict(
        expected_contract_hashes,
    ):
        raise ValueError("Branch checkpoint contract hashes differ")
    rng_value = payload["rng_state"]
    if not isinstance(rng_value, dict):
        raise TypeError("Branch checkpoint RNG state has an unexpected type")
    _rng_from_payload(cast("Mapping[str, object]", rng_value))
    for state_key in (
        "model_state_dict",
        "optimizer_state_dict",
        "grad_scaler_state_dict",
    ):
        if not isinstance(payload[state_key], dict):
            raise TypeError(f"Branch checkpoint {state_key} must be a mapping")


def save_branch_checkpoint(
    checkpoints_root: Path,
    *,
    slot: str,
    payload: Mapping[str, object],
) -> LoadedBranchCheckpoint:
    """Durably publish an immutable manifest before atomically moving a slot pointer."""
    _validate_checkpoint_slot(slot)
    validate_branch_checkpoint_payload(payload)
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    objects_root = checkpoints_root / ".objects"
    objects_root.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=objects_root))
    try:
        checkpoint_path = temporary / "checkpoint.pt"
        torch.save(dict(payload), checkpoint_path)
        _fsync_file(checkpoint_path)
        checkpoint_sha256 = _sha256(checkpoint_path)
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA,
            "branch_name": payload["branch_name"],
            "committed_update": payload["committed_update"],
            "checkpoint_sha256": checkpoint_sha256,
        }
        metadata_path = temporary / "metadata.json"
        metadata_path.write_bytes(_canonical_json(metadata))
        _fsync_file(metadata_path)
        manifest = {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
            "files": {
                "checkpoint.pt": checkpoint_sha256,
                "metadata.json": _sha256(metadata_path),
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        _fsync_file(manifest_path)
        manifest_sha256 = _sha256(manifest_path)
        object_name = manifest_sha256
        destination = objects_root / object_name
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            temporary.replace(destination)
            _fsync_directory(objects_root)
        pointer = {
            "schema_version": CHECKPOINT_POINTER_SCHEMA,
            "slot": slot,
            "manifest": f".objects/{object_name}/manifest.json",
            "manifest_sha256": manifest_sha256,
        }
        pointer_path = checkpoints_root / slot
        temporary_pointer = checkpoints_root / f".{slot}.tmp"
        temporary_pointer.write_bytes(_canonical_json(pointer))
        _fsync_file(temporary_pointer)
        Path(temporary_pointer).replace(pointer_path)
        _fsync_directory(checkpoints_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        (checkpoints_root / f".{slot}.tmp").unlink(missing_ok=True)
        raise
    return load_branch_checkpoint(checkpoints_root, slot=slot)


def load_branch_checkpoint(
    checkpoints_root: Path,
    *,
    slot: str,
    expected_branch_name: str | None = None,
    expected_contract_hashes: Mapping[str, str] | None = None,
) -> LoadedBranchCheckpoint:
    """Verify pointer, manifest, every file hash, and payload before resume use."""
    _validate_checkpoint_slot(slot)
    checkpoints_root = checkpoints_root.resolve()
    pointer_path = checkpoints_root / slot
    if pointer_path.is_symlink():
        raise ValueError("Branch checkpoint pointer may not be a symlink")
    pointer = _read_json_object(pointer_path)
    if set(pointer) != {"schema_version", "slot", "manifest", "manifest_sha256"}:
        raise ValueError("Branch checkpoint pointer is incomplete")
    if (
        pointer["schema_version"] != CHECKPOINT_POINTER_SCHEMA
        or pointer["slot"] != slot
    ):
        raise ValueError("Branch checkpoint pointer identity differs")
    manifest_relative = pointer["manifest"]
    if not isinstance(manifest_relative, str):
        raise TypeError("Branch checkpoint manifest path must be text")
    relative = Path(manifest_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Branch checkpoint manifest escapes the object store")
    unresolved_manifest = checkpoints_root / relative
    if unresolved_manifest.is_symlink():
        raise ValueError("Branch checkpoint manifest may not be a symlink")
    manifest_path = unresolved_manifest.resolve()
    if (
        manifest_path.is_symlink()
        or manifest_path.parent.parent != checkpoints_root / ".objects"
        or not manifest_path.is_relative_to(checkpoints_root)
    ):
        raise ValueError("Branch checkpoint manifest escapes the object store")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != pointer["manifest_sha256"]:
        raise ValueError("Branch checkpoint manifest hash differs")
    manifest = _read_json_object(manifest_path)
    if (
        set(manifest) != {"schema_version", "files"}
        or manifest["schema_version"] != CHECKPOINT_MANIFEST_SCHEMA
    ):
        raise ValueError("Branch checkpoint manifest is incomplete")
    files_value = manifest["files"]
    if not isinstance(files_value, dict):
        raise TypeError("Branch checkpoint manifest files must be a mapping")
    files = cast("dict[str, object]", files_value)
    if set(files) != {"checkpoint.pt", "metadata.json"}:
        raise ValueError("Branch checkpoint manifest file set differs")
    for name, expected_sha256 in files.items():
        if not isinstance(expected_sha256, str):
            raise TypeError("Branch checkpoint manifest hashes must be text")
        file_path = manifest_path.parent / name
        if file_path.is_symlink() or file_path.resolve().parent != manifest_path.parent:
            raise ValueError(f"Branch checkpoint file escapes its object: {name}")
        if _sha256(file_path) != expected_sha256:
            raise ValueError(f"Branch checkpoint file hash differs: {name}")
    checkpoint_path = manifest_path.parent / "checkpoint.pt"
    payload_value = cast(
        "object",
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
    )
    if not isinstance(payload_value, dict):
        raise TypeError("Branch checkpoint payload must be a mapping")
    payload = cast("dict[str, object]", payload_value)
    validate_branch_checkpoint_payload(
        payload,
        expected_branch_name=expected_branch_name,
        expected_contract_hashes=expected_contract_hashes,
    )
    metadata = _read_json_object(manifest_path.parent / "metadata.json")
    if metadata != {
        "schema_version": CHECKPOINT_SCHEMA,
        "branch_name": payload["branch_name"],
        "committed_update": payload["committed_update"],
        "checkpoint_sha256": files["checkpoint.pt"],
    }:
        raise ValueError("Branch checkpoint metadata differs from its payload")
    return LoadedBranchCheckpoint(
        payload=payload,
        checkpoint_sha256=cast("str", files["checkpoint.pt"]),
        manifest_sha256=manifest_sha256,
        slot=slot,
    )


def restore_branch_checkpoint(
    loaded: LoadedBranchCheckpoint,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: StepScaler,
) -> tuple[int, int, int]:
    """Restore a validated branch and return committed update, epoch, and cursor."""
    payload = loaded.payload
    model.load_state_dict(cast("dict[str, Tensor]", payload["model_state_dict"]))
    optimizer.load_state_dict(
        cast("dict[str, object]", payload["optimizer_state_dict"]),
    )
    scaler.load_state_dict(
        cast("dict[str, object]", payload["grad_scaler_state_dict"]),
    )
    restore_rng_state(
        _rng_from_payload(cast("Mapping[str, object]", payload["rng_state"])),
    )
    return (
        cast("int", payload["committed_update"]),
        cast("int", payload["epoch"]),
        cast("int", payload["within_epoch_cursor"]),
    )


def _selection_payload(state: BestSelectionState) -> dict[str, object]:
    return {
        "best_metrics": state.best_metrics.to_dict() if state.best_metrics else None,
        "best_boundary": state.best_boundary,
        "non_improving_checks": state.non_improving_checks,
        "stopped": state.stopped,
    }


def selection_from_checkpoint_payload(
    value: Mapping[str, object],
) -> BestSelectionState:
    """Decode the safe primitive selector representation after validation."""
    return _selection_from_payload(value)


def _selection_from_payload(value: Mapping[str, object]) -> BestSelectionState:
    if set(value) != {
        "best_metrics",
        "best_boundary",
        "non_improving_checks",
        "stopped",
    }:
        raise ValueError("Checkpoint selector fields differ")
    metrics_value = value["best_metrics"]
    metrics = None
    if metrics_value is not None:
        if not isinstance(metrics_value, dict):
            raise TypeError("Checkpoint best metrics must be an object")
        metrics = _validation_metrics_from_payload(
            cast("Mapping[str, object]", metrics_value),
        )
    boundary = value["best_boundary"]
    non_improving = value["non_improving_checks"]
    stopped = value["stopped"]
    if boundary is not None and (
        not isinstance(boundary, int) or isinstance(boundary, bool)
    ):
        raise TypeError("Checkpoint best boundary must be an integer or null")
    if (
        not isinstance(non_improving, int)
        or isinstance(non_improving, bool)
        or non_improving < 0
    ):
        raise TypeError("Checkpoint non-improving count must be a nonnegative integer")
    if not isinstance(stopped, bool):
        raise TypeError("Checkpoint stopped flag must be boolean")
    if (metrics is None) != (boundary is None):
        raise ValueError("Checkpoint best metrics and boundary must appear together")
    return BestSelectionState(
        best_metrics=metrics,
        best_boundary=boundary,
        non_improving_checks=non_improving,
        stopped=stopped,
    )


def _validation_metrics_from_payload(value: Mapping[str, object]) -> ValidationMetrics:
    expected_fields = {
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "mean_ce",
        "per_class",
        "confusion_matrix",
        "wsi_count",
    }
    if set(value) != expected_fields:
        raise ValueError("Checkpoint validation metric fields differ")
    scalar_names = ("macro_f1", "balanced_accuracy", "accuracy", "mean_ce")
    scalars: dict[str, float] = {}
    for name in scalar_names:
        raw_scalar = value[name]
        if not isinstance(raw_scalar, float) or not math.isfinite(raw_scalar):
            raise TypeError(
                f"Checkpoint validation metric {name} must be a finite float",
            )
        scalars[name] = raw_scalar
    wsi_count = value["wsi_count"]
    if not isinstance(wsi_count, int) or isinstance(wsi_count, bool) or wsi_count < 1:
        raise TypeError("Checkpoint validation WSI count must be a positive integer")
    per_class_value = value["per_class"]
    confusion_value = value["confusion_matrix"]
    if not isinstance(per_class_value, (tuple, list)):
        raise TypeError("Checkpoint per-class metrics differ")
    per_class_rows = cast("Sequence[object]", per_class_value)
    if len(per_class_rows) != len(CLASS_ORDER):
        raise TypeError("Checkpoint per-class metrics differ")
    if not isinstance(confusion_value, (tuple, list)):
        raise TypeError("Checkpoint confusion matrix differs")
    confusion_rows = cast("Sequence[object]", confusion_value)
    if len(confusion_rows) != len(CLASS_ORDER):
        raise TypeError("Checkpoint confusion matrix differs")
    per_class: list[ClassMetrics] = []
    for expected_diagnosis, item_value in zip(
        CLASS_ORDER,
        per_class_rows,
        strict=True,
    ):
        if not isinstance(item_value, dict):
            raise TypeError("Checkpoint per-class metric row must be an object")
        item = cast("Mapping[str, object]", item_value)
        if set(item) != {"diagnosis", "precision", "recall", "f1", "support"}:
            raise ValueError("Checkpoint per-class metric fields differ")
        support = item["support"]
        values = (item["precision"], item["recall"], item["f1"])
        if item["diagnosis"] != expected_diagnosis:
            raise ValueError("Checkpoint per-class diagnosis order differs")
        if not isinstance(support, int) or isinstance(support, bool) or support < 0:
            raise TypeError(
                "Checkpoint per-class support must be a nonnegative integer",
            )
        if any(
            not isinstance(metric, float) or not math.isfinite(metric)
            for metric in values
        ):
            raise TypeError("Checkpoint per-class metrics must be finite floats")
        per_class.append(
            ClassMetrics(
                diagnosis=expected_diagnosis,
                precision=cast("float", values[0]),
                recall=cast("float", values[1]),
                f1=cast("float", values[2]),
                support=support,
            ),
        )
    confusion: list[tuple[int, ...]] = []
    for row_value in confusion_rows:
        if not isinstance(row_value, (tuple, list)):
            raise TypeError("Checkpoint confusion-matrix row differs")
        row_values = cast("Sequence[object]", row_value)
        if len(row_values) != len(CLASS_ORDER):
            raise TypeError("Checkpoint confusion-matrix row differs")
        row: list[int] = []
        for count in row_values:
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise TypeError(
                    "Checkpoint confusion counts must be nonnegative integers",
                )
            row.append(count)
        confusion.append(tuple(row))
    metrics = ValidationMetrics(
        macro_f1=scalars["macro_f1"],
        balanced_accuracy=scalars["balanced_accuracy"],
        accuracy=scalars["accuracy"],
        mean_ce=scalars["mean_ce"],
        per_class=tuple(per_class),
        confusion_matrix=tuple(confusion),
        wsi_count=wsi_count,
    )
    _validate_finite_metrics(metrics)
    if sum(item.support for item in metrics.per_class) != metrics.wsi_count or any(
        sum(row) != item.support
        for row, item in zip(metrics.confusion_matrix, metrics.per_class, strict=True)
    ):
        raise ValueError("Checkpoint validation supports differ from confusion counts")
    return metrics


def _rng_payload(state: RNGState) -> dict[str, object]:
    numpy_state = cast("tuple[object, ...]", state.numpy)
    return {
        "python": state.python,
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                cast("np.ndarray[Any, Any]", numpy_state[1]).copy(),
            ),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": state.torch_cpu,
        "torch_cuda": state.torch_cuda,
    }


def _rng_from_payload(value: Mapping[str, object]) -> RNGState:
    if set(value) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Checkpoint RNG fields differ")
    numpy_value = value["numpy"]
    if not isinstance(numpy_value, dict):
        raise TypeError("Checkpoint NumPy RNG state must be an object")
    raw = cast("Mapping[str, object]", numpy_value)
    if set(raw) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("Checkpoint NumPy RNG fields differ")
    keys = raw["keys"]
    cpu = value["torch_cpu"]
    cuda = value["torch_cuda"]
    if not isinstance(cuda, (tuple, list)):
        raise TypeError("Checkpoint RNG tensors differ")
    cuda_values = cast("Sequence[object]", cuda)
    if (
        not isinstance(keys, Tensor)
        or not isinstance(cpu, Tensor)
        or any(not isinstance(item, Tensor) for item in cuda_values)
    ):
        raise TypeError("Checkpoint RNG tensors differ")
    algorithm = raw["algorithm"]
    position = raw["position"]
    has_gauss = raw["has_gauss"]
    cached_gaussian = raw["cached_gaussian"]
    if algorithm != "MT19937":
        raise ValueError("Checkpoint NumPy RNG algorithm differs")
    if (
        keys.dtype != torch.uint32
        or keys.device.type != "cpu"
        or tuple(keys.shape) != (624,)
        or not isinstance(position, int)
        or isinstance(position, bool)
        or not 0 <= position <= 624
        or not isinstance(has_gauss, int)
        or isinstance(has_gauss, bool)
        or has_gauss not in {0, 1}
        or not isinstance(cached_gaussian, float)
        or not math.isfinite(cached_gaussian)
    ):
        raise ValueError("Checkpoint NumPy RNG values differ")
    cuda_tensors = cast("Sequence[Tensor]", cuda_values)
    rng_tensors = (cpu, *cuda_tensors)
    if any(
        tensor.dtype != torch.uint8
        or tensor.device.type != "cpu"
        or tensor.ndim != 1
        or tensor.numel() < 1
        for tensor in rng_tensors
    ):
        raise ValueError("Checkpoint Torch RNG tensor values differ")
    python_state = value["python"]
    try:
        random.Random().setstate(cast("tuple[Any, ...]", copy.deepcopy(python_state)))
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint Python RNG state differs") from error
    return RNGState(
        python=python_state,
        numpy=(
            algorithm,
            keys.cpu().numpy().copy(),
            position,
            has_gauss,
            cached_gaussian,
        ),
        torch_cpu=cpu,
        torch_cuda=tuple(cuda_tensors),
    )


def _selection_key(
    metrics: ValidationMetrics,
    boundary: int,
) -> tuple[float, float, int]:
    return metrics.macro_f1, -metrics.mean_ce, -boundary


def _validate_finite_metrics(metrics: ValidationMetrics) -> None:
    scalars = (
        metrics.macro_f1,
        metrics.balanced_accuracy,
        metrics.accuracy,
        metrics.mean_ce,
    )
    if metrics.wsi_count < 1 or any(not math.isfinite(value) for value in scalars):
        raise ValueError("Selector metrics must be finite and nonempty")


def _diagnosis_strata(truths: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    class_count = len(CLASS_ORDER)
    if any(not 0 <= truth < class_count for truth in truths):
        raise ValueError("Bootstrap truths must use canonical class indices")
    strata = tuple(
        tuple(index for index, truth in enumerate(truths) if truth == class_index)
        for class_index in range(class_count)
    )
    if any(not stratum for stratum in strata):
        raise ValueError("Diagnosis-stratified bootstrap requires every class")
    return strata


def _named_nonfinite_gradients(model: nn.Module) -> list[dict[str, object]]:
    """Materialize per-parameter diagnostics only after GradScaler backs off."""
    nonfinite: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all().item()):
            nonfinite.append(
                {
                    "name": name,
                    "nonfinite_count": int((~finite).sum().item()),
                },
            )
    return nonfinite


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    state = cast("Mapping[str, Tensor]", model.state_dict())
    return {name: value.detach().clone() for name, value in state.items()}


def _model_state_equal(model: nn.Module, expected: Mapping[str, Tensor]) -> bool:
    actual = cast("Mapping[str, Tensor]", model.state_dict())
    return set(actual) == set(expected) and all(
        torch.equal(actual[name], expected[name]) for name in actual
    )


def _order_sha256(order: Sequence[object]) -> str:
    return hashlib.sha256(_canonical_json({"order": list(order)})).hexdigest()


def _validate_checkpoint_slot(slot: str) -> None:
    if slot not in CHECKPOINT_SLOTS:
        raise ValueError(f"Checkpoint slot must be one of {CHECKPOINT_SLOTS!r}")


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n".encode()


def _read_json_object(path: Path) -> dict[str, object]:
    value = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return cast("dict[str, object]", value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ADAMW_BETAS",
    "ADAMW_EPSILON",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BRANCH_NAMES",
    "CALIBRATION_DIAGNOSES",
    "CALIBRATION_WSI_IDS",
    "CHECKPOINT_SLOTS",
    "CLASS_WEIGHTS",
    "CLASS_WEIGHT_BY_DIAGNOSIS",
    "CLASS_WEIGHT_BY_INDEX",
    "EARLY_STOPPING_START_EPOCH",
    "EARLY_STOPPING_START_UPDATE",
    "MAXIMUM_EPOCHS",
    "MAX_SCALE_BACKOFFS",
    "MIL_TRAINING_CONFIG",
    "PATIENCE_CHECKS",
    "PATIENCE_EPOCHS",
    "PEAK_LEARNING_RATE",
    "SCALER_BACKOFF_FACTOR",
    "SCALER_GROWTH_FACTOR",
    "SCALER_GROWTH_INTERVAL",
    "SCALER_INITIAL_SCALE",
    "TOTAL_UPDATES",
    "TRAINING_SEED",
    "TRAIN_CLASS_COUNTS",
    "TRAIN_WSI_COUNT",
    "WARMUP_EPOCHS",
    "WARMUP_START_LEARNING_RATE",
    "WARMUP_START_RATIO",
    "WARMUP_UPDATES",
    "BestSelectionState",
    "BootstrapInterval",
    "BranchNumericalError",
    "CalibrationCaseResult",
    "CalibrationResult",
    "ClassMetrics",
    "LoadedBranchCheckpoint",
    "MILTrainingConfig",
    "RNGState",
    "SelectionUpdate",
    "StepScaler",
    "ValidationMetrics",
    "WeightedAMPStepResult",
    "branch_checkpoint_payload",
    "calibrate_grad_scaler",
    "capture_rng_state",
    "class_weight_for_label",
    "committed_update_learning_rate",
    "compute_validation_metrics",
    "diagnosis_stratified_paired_bootstrap",
    "is_validation_boundary",
    "load_branch_checkpoint",
    "make_default_grad_scaler",
    "make_fused_adamw",
    "predictions_from_logits",
    "restore_branch_checkpoint",
    "restore_rng_state",
    "save_branch_checkpoint",
    "selection_from_checkpoint_payload",
    "training_epoch_order",
    "update_best_selection",
    "validate_branch_checkpoint_payload",
    "validate_default_grad_scaler",
    "weighted_amp_step",
]
