# Copyright 2026 HiperMaximus
"""Non-promotable capped real-data runtime pretest artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import math
import os
import subprocess  # noqa: S404
import sys
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

import torch

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.benchmarking.runtime_schema import (
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
)
from eqvae.config import ResolvedConfig, resolve_json_config
from eqvae.data.roots import REAL_TRAIN_PATCH_COUNT
from eqvae.models.activations import GatedScalarActivation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from torch.utils.hooks import RemovableHandle

    from eqvae.corruption.stain import StainCorruptionProfile, StainCorruptionResult
    from eqvae.data.patch_shards import PatchRecord
    from eqvae.data.roots import PatchDataPaths, PatchSplitPaths
    from eqvae.data.training_batches import PatchTrainingBatch
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE
    from eqvae.training.fastpath_step import FastpathStepOutput
    from eqvae.training.step import TrainStepRequest, TrainStepResult

    type TensorPayload = dict[str, torch.Tensor]

REAL_DATA_PRETEST_SCHEMA_VERSION = "spec0001.real_data_runtime_pretest.v1"
REAL_DATA_PRETEST_KIND = "real_data_runtime_pretest"
REAL_DATA_PRETEST_SOURCE = "kaggle_capped_real_data_train_step_pretest"
REAL_DATA_PRETEST_SCOPE = "non_promotable_real_data_runtime_pretest"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_REAL_TRAIN_PATCH_COUNT = REAL_TRAIN_PATCH_COUNT
EXPECTED_REAL_VALIDATION_PATCH_COUNT = 30_000
EXPECTED_REAL_TRAIN_WSI_COUNT = 322
EXPECTED_REAL_VALIDATION_WSI_COUNT = 39
EXPECTED_CAP_TRAIN_PATCH_COUNT = 8_192
EXPECTED_CAP_VALIDATION_PATCH_COUNT = 2_048
EXPECTED_WINDOW_POLICY = "fixed_hashed_spread_windows"
EXPECTED_TRAIN_WINDOWS = (
    ("train_head", 0, 2_048),
    ("train_mid_a", 98_304, 2_048),
    ("train_mid_b", 196_608, 2_048),
    ("train_tail", 297_952, 2_048),
)
EXPECTED_VALIDATION_WINDOWS = (
    ("validation_head", 0, 1_024),
    ("validation_tail", 28_976, 1_024),
)
MANIFEST_FILENAME = "real_data_runtime_pretest_manifest.json"
RUNTIME_PROOF_FILENAME = "runtime_proof.json"
RUNTIME_MATRIX_FILENAME = "runtime_matrix.csv"
DATALOADER_MATRIX_FILENAME = "dataloader_matrix.csv"
NUMERICAL_CHECKS_FILENAME = "numerical_checks.csv"
CORRUPTION_CHECKS_FILENAME = "corruption_checks.csv"
GATE_HEALTH_FILENAME = "gate_health.csv"
GATE_HEALTH_SUMMARY_FILENAME = "gate_health_summary.json"
RECOMMENDATIONS_FILENAME = "real_data_runtime_pretest_recommendations.json"
PHASE_TIMINGS_FILENAME = "phase_timings.json"
SELECTED_RUNTIME_FILENAME = "selected_runtime.json"
SINGLE_VISIBLE_T4 = "single_visible_t4"
DUAL_T4_DDP = "dual_t4_ddp"
AMP_OFF_FP32 = "amp_off_fp32"
COMPILE_NONE = "none"
COMPILE_MODEL_FORWARD = "model_forward"
COMPILE_STEP = "step"
BRANCHLESS_ALL = "branchless_all"
INDEXED_MASKED = "indexed_masked"
PASS_STATUS = "pass"  # noqa: S105
LOCAL_PASS_STATUS = "local_pass"  # noqa: S105
FAIL_STATUS = "fail"
INELIGIBLE_STATUS = "ineligible"
SKIPPED_UNSUPPORTED = "skipped_unsupported"
WRONG_ACCELERATOR = "wrong_accelerator"
RUNTIME_ERROR = "runtime_error"
DUAL_T4_DEVICE_COUNT = 2
DEFAULT_DATALOADER_NUM_WORKERS = 0
DEFAULT_DATALOADER_PREFETCH_FACTOR = ""
DEFAULT_DATALOADER_PIN_MEMORY = False
DEFAULT_DATALOADER_PERSISTENT_WORKERS = False
DEFAULT_DATALOADER_NON_BLOCKING_H2D = True
MIN_LOADER_TRAINER_THROUGHPUT_RATIO = 1.25
ADAM_BETA_COUNT = 2
LATENT_DOWNSAMPLE_FACTOR = 8
MODEL_INPUT_SHAPE_NDIM = 4
FILE_HASH_CHUNK_BYTES = 8 * 1024 * 1024
VALIDATION_CLEAN_PROOF_BATCH_SIZE = 12
LOCAL_LINKED_EVIDENCE_BATCH_SIZE = 2
REQUIRED_NUMERICAL_FIXED_BATCHES = 3
REQUIRED_COMPILE_SETTLE_STEPS = 5
DATA_ROOT_RESOLUTION_ATTEMPTS = 4
DATA_ROOT_RETRY_SLEEP_SEC = 5.0
MAX_DATA_WAIT_FRACTION = 0.20
MIN_CHANNEL_TENSOR_NDIM = 2
MAX_GATE_QUANTILE_ELEMENTS = 1_000_000
GATE_SATURATION_LOW = 0.01
GATE_SATURATION_HIGH = 0.99
GATE_DEAD_RMS_THRESHOLD = 1.0e-8
GATE_WARNING_ABS_PARAM = 10.0
GATE_FAILURE_ABS_PARAM = 20.0
GATE_WARNING_SATURATION_FRACTION = 0.80
GATE_FAILURE_SATURATION_FRACTION = 0.95
GATE_WARNING_OUTPUT_INPUT_RATIO = 1.0e-2
GATE_FAILURE_OUTPUT_INPUT_RATIO = 1.0e-3
NUMERICAL_ABS_THRESHOLD = 1.0e-3
NUMERICAL_REL_THRESHOLD = 5.0e-3
NUMERICAL_KL_REL_THRESHOLD = 1.0e-2
NUMERICAL_NORM_REL_THRESHOLD = 5.0e-2


class NormalizeUint8BatchFn(Protocol):
    """Typed local-import callable for uint8 normalization."""

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Normalize one uint8 batch."""
        ...


class CorruptNormalizedBatchFn(Protocol):
    """Typed local-import callable for corruption dispatch."""

    def __call__(  # noqa: PLR0913
        self,
        images: torch.Tensor,
        *,
        profile: StainCorruptionProfile,
        corruption_seed: int,
        split: str,
        semantic_sample_keys: Sequence[str],
        corruption_step: int,
        corruption_view: str,
        strategy: str,
    ) -> StainCorruptionResult:
        """Corrupt one normalized batch."""
        ...


class DataRootUnavailableError(FileNotFoundError):
    """Data root resolution failed with JSON-safe diagnostics attached."""

    diagnostics: JsonObject

    def __init__(self, message: str, *, diagnostics: JsonObject) -> None:
        """Initialize the resolution failure."""
        super().__init__(message)
        self.diagnostics = diagnostics


class BetaForStepFn(Protocol):
    """Typed local-import callable for beta scheduling."""

    def __call__(
        self,
        *,
        optimizer_step_index: int,
        max_optimizer_steps: int,
        target_beta: float,
        warmup_fraction: float,
    ) -> float:
        """Return beta for one step."""
        ...


class TrainStepRequestFactory(Protocol):
    """Typed constructor protocol for `TrainStepRequest`."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        model: NonEquivariantVAE,
        optimizer: torch.optim.Optimizer,
        clean_batch: torch.Tensor,
        eps: torch.Tensor,
        beta: float,
        ssim_weight: float,
        optimizer_step_index: int,
        gradient_clip_global_norm: float,
        input_batch: torch.Tensor | None,
    ) -> TrainStepRequest:
        """Build one train-step request."""
        ...


class TrainStepRunner(Protocol):
    """Typed callable for the local train-step helper."""

    def __call__(self, request: TrainStepRequest) -> TrainStepResult:
        """Run one train step."""
        ...


class CudaDeviceProperties(Protocol):
    """Typed subset of CUDA device properties used by the pretest."""

    total_memory: int


class CudaDevicePropertiesGetter(Protocol):
    """Typed CUDA device-properties lookup."""

    def __call__(self, device: torch.device) -> CudaDeviceProperties:
        """Return properties for one CUDA device."""
        ...


@dataclass(frozen=True)
class RealDataRuntimePretestRequest:
    """Inputs for a capped real-data runtime pretest."""

    config_path: Path
    output_dir: Path
    data_root: str | None = None


@dataclass(frozen=True)
class RealDataRuntimePretestSettings:
    """Resolved pretest settings from config."""

    run_name: str
    dataset_slug: str
    data_root: str
    image_size: int
    channels: int
    real_train_patch_count: int
    real_validation_patch_count: int
    cap_train_patch_count: int
    cap_validation_patch_count: int
    window_policy: str
    train_windows: tuple[WindowSpec, ...]
    validation_windows: tuple[WindowSpec, ...]
    warmup_steps: int
    measured_steps: int
    repeats: int
    compile_settle_steps: int
    compile_scopes: tuple[str, ...]
    corruption_strategies: tuple[str, ...]
    seeded_candidates: tuple[SeededCandidate, ...]
    blocked_claims: JsonObject
    ssim_weight: float
    beta_target: float
    beta_warmup_fraction: float
    learning_rate: float
    weight_decay: float
    gradient_clip_global_norm: float
    global_seed: int
    data_seed: int
    corruption_seed: int
    latent_seed: int
    corruption_config: JsonObject
    norm_groups: int


@dataclass(frozen=True)
class WindowSpec:
    """One fixed spread window from the pre-shuffled shard."""

    name: str
    start_row: int
    patch_count: int

    @property
    def stop_row(self) -> int:
        """Return exclusive stop row."""
        return self.start_row + self.patch_count


@dataclass(frozen=True)
class SeededCandidate:
    """Accelerator/batch candidate with synthetic-v4 provenance."""

    accelerator_mode: str
    per_device_batch_size: int
    synthetic_v4_rank: int | None
    synthetic_v4_row_id: str | None
    candidate_role: str


@dataclass(frozen=True)
class RowSpec:
    """One runtime-matrix row attempt."""

    row_id: str
    accelerator_mode: str
    per_device_batch_size: int
    precision_policy: str
    compile_scope: str
    corruption_strategy: str
    parent_synthetic_row_id: str
    candidate_role: str
    world_size: int
    nproc_per_node: int
    cuda_visible_devices: str
    runtime_policy_id: str = "fp32_eager_default"
    memory_format: str = "contiguous"
    autocast_dtype: str = ""
    fp32_loss: bool = True
    grad_scaler_enabled: bool = False
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    deterministic_algorithms: bool = False
    tf32_enabled: bool = False
    matmul_precision: str = "highest"
    ddp_static_graph: bool = False
    ddp_gradient_as_bucket_view: bool = False
    optimizer_implementation: str = "adamw_default"
    zero_grad_set_to_none: bool = True
    gradient_clip_foreach: bool = False
    compile_dynamic: bool = False
    # Spec 0011 S14a: compiled fast-path recipe knobs measured by the efficiency
    # search. Eager-v5 defaults keep every existing (eager) row byte-identical; a
    # compiled winner row overrides them from its policy (see the executor RowSpec).
    optimize_ddp: str = ""
    compiled_autograd: bool = False
    reorder_compute_comm_overlap: bool = False
    ddp_broadcast_buffers: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int | None = None
    fused_optimizer: bool = False


@dataclass(frozen=True)
class ChildRowConfig:
    """Serializable child-process row configuration."""

    config_path: Path
    output_dir: Path
    data_root: str
    row_spec: RowSpec
    settings: RealDataRuntimePretestSettings


@dataclass(frozen=True)
class ChildProcessArgs:
    """CLI args for child row execution."""

    child_row: str | None
    ddp_rank_row: str | None


class _CandidateTrainStepEvidenceError(RuntimeError):
    """Strategy-scoped train-step evidence failure without retaining tensors."""

    def __init__(
        self,
        *,
        strategy_attempt: str,
        target_corruption_strategy: str,
        cause: BaseException,
    ) -> None:
        self.strategy_attempt = strategy_attempt
        self.target_corruption_strategy = target_corruption_strategy
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)
        super().__init__(self.cause_message)


class PhaseTimingRecorder:
    """Record coarse pretest phase timings and emit progress logs."""

    def __init__(self) -> None:
        self._started_ns = time.perf_counter_ns()
        self._started_at_utc = _utc_timestamp()
        self._records: list[JsonObject] = []

    def phase(self, name: str, **metadata: JsonValue) -> _PhaseTimingContext:
        """Create a context manager that records one named phase.

        Returns:
            Context manager for the requested phase.

        """
        return _PhaseTimingContext(
            recorder=self,
            name=name,
            metadata=dict(metadata),
        )

    def payload(self) -> JsonObject:
        """Return the serializable phase-timing summary.

        Returns:
            JSON-ready phase timing summary.

        """
        return {
            "schema_version": "eqvae.phase_timings.v1",
            "started_at_utc": self._started_at_utc,
            "finished_at_utc": _utc_timestamp(),
            "total_elapsed_sec": _rounded_float(_elapsed_seconds(self._started_ns)),
            "recorded_phase_count": len(self._records),
            "polling_hint": (
                "Use artifact phase timings to choose future polling cadence; "
                "source-attached real-data kernels should default to 30-minute "
                "or slower status polling until terminal durations are known."
            ),
            "phases": [dict(record) for record in self._records],
        }

    def record(self, record: JsonObject) -> None:
        """Store one completed phase timing record."""
        self._records.append(record)


@dataclass
class _PhaseTimingContext:
    recorder: PhaseTimingRecorder
    name: str
    metadata: JsonObject
    started_at_utc: str = ""
    start_ns: int = 0

    def __enter__(self) -> Self:
        self.started_at_utc = _utc_timestamp()
        self.start_ns = time.perf_counter_ns()
        _log_phase_event(
            _phase_log_payload(
                event="start",
                name=self.name,
                metadata=self.metadata,
            ),
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc
        del traceback
        status = FAIL_STATUS if exc_type is not None else PASS_STATUS
        failure_type = exc_type.__name__ if exc_type is not None else ""
        elapsed_sec = _elapsed_seconds(self.start_ns)
        record: JsonObject = {
            "name": self.name,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": _utc_timestamp(),
            "elapsed_sec": _rounded_float(elapsed_sec),
        }
        if self.metadata:
            record["metadata"] = self.metadata
        if failure_type:
            record["failure_type"] = failure_type
        self.recorder.record(record)
        extra: JsonObject = {
            "status": status,
            "elapsed_sec": _rounded_float(elapsed_sec),
        }
        if failure_type:
            extra["failure_type"] = failure_type
        _log_phase_event(
            _phase_log_payload(
                event="finish",
                name=self.name,
                metadata=self.metadata,
                extra=extra,
            ),
        )
        return False


def write_real_data_runtime_pretest(request: RealDataRuntimePretestRequest) -> Path:
    """Run the capped pretest surface and write non-promotable artifacts.

    Returns:
        Path to `benchmark/real_data_runtime_pretest_recommendations.json`.

    """
    phase_timings = PhaseTimingRecorder()
    with phase_timings.phase(
        "config_resolution",
        config_path=str(request.config_path),
        output_dir=str(request.output_dir),
    ):
        resolved = resolve_json_config(request.config_path)
        settings = _settings(resolved, data_root_override=request.data_root)
    benchmark_dir = request.output_dir / "benchmark"
    metrics_dir = request.output_dir / "metrics"
    with phase_timings.phase("prepare_output_dir", output_dir=str(request.output_dir)):
        request.output_dir.mkdir(parents=True, exist_ok=True)
        _reject_selected_runtime_artifact(request.output_dir)
    with phase_timings.phase(
        "real_data_identity_and_clean_path_proof",
        data_root=settings.data_root,
        dataset_slug=settings.dataset_slug,
    ):
        data_proof = _real_data_identity_and_clean_path_proof(settings)
    row_specs = _stage1_row_specs(settings)
    with phase_timings.phase(
        "stage1_runtime_rows",
        configured_row_count=len(row_specs),
    ):
        rows = _run_stage1_rows(
            request=request,
            settings=settings,
            row_specs=row_specs,
            phase_timings=phase_timings,
        )
    with phase_timings.phase("linked_evidence_payload", row_count=len(rows)):
        linked_evidence = _linked_evidence_payload(
            request=request,
            settings=settings,
            data_proof=data_proof,
            rows=rows,
            phase_timings=phase_timings,
        )
    with phase_timings.phase("row_linked_evidence_join", row_count=len(rows)):
        rows = _rows_with_linked_evidence(
            rows=rows,
            data_proof=data_proof,
            linked_evidence=linked_evidence,
        )
    with phase_timings.phase("schema_linked_rows", row_count=len(rows)):
        dataloader_rows = _schema_dataloader_rows(
            settings=settings,
            data_proof=data_proof,
            linked_evidence=linked_evidence,
        )
        numerical_rows = _schema_numerical_rows(
            settings=settings,
            rows=rows,
            linked_evidence=linked_evidence,
        )
        corruption_rows = _schema_corruption_rows(
            settings=settings,
            rows=rows,
            linked_evidence=linked_evidence,
        )
        gate_health_rows = _gate_health_rows(
            settings=settings,
            linked_evidence=linked_evidence,
        )

    with phase_timings.phase("write_artifacts", benchmark_dir=str(benchmark_dir)):
        write_csv(benchmark_dir / RUNTIME_MATRIX_FILENAME, RUNTIME_MATRIX_COLUMNS, rows)
        write_csv(
            benchmark_dir / DATALOADER_MATRIX_FILENAME,
            DATALOADER_MATRIX_COLUMNS,
            dataloader_rows,
        )
        write_csv(
            benchmark_dir / NUMERICAL_CHECKS_FILENAME,
            NUMERICAL_CHECK_COLUMNS,
            numerical_rows,
        )
        write_csv(
            benchmark_dir / CORRUPTION_CHECKS_FILENAME,
            CORRUPTION_CHECK_COLUMNS,
            corruption_rows,
        )
        write_csv(
            metrics_dir / GATE_HEALTH_FILENAME,
            GATE_HEALTH_COLUMNS,
            gate_health_rows,
        )
        write_json(
            benchmark_dir / GATE_HEALTH_SUMMARY_FILENAME,
            _gate_health_summary_payload(linked_evidence=linked_evidence),
        )
        recommendations_path = benchmark_dir / RECOMMENDATIONS_FILENAME
        write_json(
            recommendations_path,
            _recommendations_payload(settings=settings, rows=rows),
        )
    final_phase_payload = phase_timings.payload()
    write_json(
        benchmark_dir / MANIFEST_FILENAME,
        _manifest_payload(
            request=request,
            resolved=resolved,
            settings=settings,
            rows=rows,
            data_proof=data_proof,
            linked_evidence=linked_evidence,
            phase_timings=final_phase_payload,
        ),
    )
    write_json(
        benchmark_dir / RUNTIME_PROOF_FILENAME,
        _runtime_proof_payload(
            settings=settings,
            rows=rows,
            data_proof=data_proof,
            linked_evidence=linked_evidence,
            phase_timings=final_phase_payload,
        ),
    )
    write_json(benchmark_dir / PHASE_TIMINGS_FILENAME, final_phase_payload)
    _reject_selected_runtime_artifact(request.output_dir)
    return recommendations_path


def write_local_upload_simulation_artifact(
    *,
    config_path: Path,
    output_dir: Path,
    payload_manifest: JsonObject,
) -> Path:
    """Write an import-only local upload simulation artifact.

    Returns:
        Path to the artifact.

    """
    resolved = resolve_json_config(config_path)
    settings = _settings(resolved, data_root_override=None)
    _reject_selected_runtime_artifact(output_dir)
    payload: JsonObject = {
        "status": "import_smoke_pass",
        "status_scope": "non_promotable_local_upload_simulation",
        "benchmark_kind": "real_data_runtime_pretest_import_only",
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": False,
        "writes_selected_runtime": False,
        "blocked_claims": settings.blocked_claims,
        "config_exists": config_path.exists(),
        "payload_manifest": payload_manifest,
        "expected_dataset_slug": EXPECTED_DATASET_SLUG,
    }
    output_path = output_dir / "benchmark" / "real_data_runtime_pretest_import.json"
    write_json(output_path, payload)
    _reject_selected_runtime_artifact(output_dir)
    return output_path


def _reject_selected_runtime_artifact(output_dir: Path) -> None:
    selected_runtime = output_dir / "benchmark" / SELECTED_RUNTIME_FILENAME
    if selected_runtime.exists():
        message = (
            "real-data runtime pretest must not leave "
            f"benchmark/{SELECTED_RUNTIME_FILENAME}"
        )
        raise RuntimeError(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run child row helper mode.

    Returns:
        Process exit code.

    Raises:
        ValueError: If called without a child-row payload.

    """
    args = _parse_args(argv)
    if args.child_row is None:
        if args.ddp_rank_row is not None:
            _run_ddp_rank_row(_decode_child_config(args.ddp_rank_row))
            return 0
        message = "Expected --child-row or --ddp-rank-row for helper mode"
        raise ValueError(message)
    _run_child_row(_decode_child_config(args.child_row))
    return 0


def _run_stage1_rows(
    *,
    request: RealDataRuntimePretestRequest,
    settings: RealDataRuntimePretestSettings,
    row_specs: Sequence[RowSpec],
    phase_timings: PhaseTimingRecorder,
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row_spec in row_specs:
        with phase_timings.phase(
            "stage1_runtime_row",
            row_id=row_spec.row_id,
            accelerator_mode=row_spec.accelerator_mode,
            compile_scope=row_spec.compile_scope,
            precision_policy=row_spec.precision_policy,
            per_device_batch_size=row_spec.per_device_batch_size,
        ):
            if row_spec.precision_policy != AMP_OFF_FP32:
                rows.append(
                    _unsupported_row(
                        settings=settings,
                        row_spec=row_spec,
                        failure_kind="amp_followup_requires_stable_fp32_candidates",
                    ),
                )
                continue
            if row_spec.compile_scope not in {
                COMPILE_NONE,
                COMPILE_MODEL_FORWARD,
                COMPILE_STEP,
            }:
                rows.append(
                    _unsupported_row(
                        settings=settings,
                        row_spec=row_spec,
                        failure_kind="compile_scope_implementation_pending",
                    ),
                )
                continue
            if row_spec.accelerator_mode != SINGLE_VISIBLE_T4:
                rows.append(
                    _unsupported_row(
                        settings=settings,
                        row_spec=row_spec,
                        failure_kind="dual_t4_ddp_train_step_measurement_pending",
                    ),
                )
                continue
            rows.append(
                _run_single_child_row(
                    ChildRowConfig(
                        config_path=request.config_path,
                        output_dir=request.output_dir,
                        data_root=settings.data_root,
                        row_spec=row_spec,
                        settings=settings,
                    ),
                ),
            )
    return rows


def _run_single_child_row(config: ChildRowConfig) -> CsvRow:
    encoded = _encode_child_config(config)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = config.row_spec.cuda_visible_devices
    environment["PYTHONPATH"] = _pythonpath_with_current_sys_path(environment)
    completed = subprocess.run(  # noqa: S603
        (
            sys.executable,
            "-m",
            "eqvae.benchmarking.real_data_runtime_pretest",
            "--child-row",
            encoded,
        ),
        cwd=config.output_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    row_path = (
        config.output_dir
        / "benchmark"
        / "child_rows"
        / f"{config.row_spec.row_id}.json"
    )
    if completed.returncode != 0 or not row_path.exists():
        return _failure_row(
            settings=config.settings,
            row_spec=config.row_spec,
            status=RUNTIME_ERROR,
            failure_kind="child_row_failed",
            failure_message=f"{completed.stderr}\n{completed.stdout}",
        )
    payload = cast("JsonObject", json.loads(row_path.read_text(encoding="utf-8")))
    row_path.unlink(missing_ok=True)
    with suppress(OSError):
        row_path.parent.rmdir()
    return _row_from_child_payload(settings=config.settings, payload=payload)


def _run_child_row(config: ChildRowConfig) -> None:  # noqa: C901, PLR0914, PLR0915
    # Torch and dependent modules are intentionally imported only in row helpers.
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    from eqvae.corruption.stain import (  # noqa: PLC0415
        corrupt_normalized_batch,
        profile_from_config,
    )
    from eqvae.data.dataloaders import normalize_uint8_batch  # noqa: PLC0415
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415
    from eqvae.data.training_batches import (  # noqa: PLC0415
        PatchTrainingDataset,
        PatchTrainingDatasetSpec,
        collate_patch_training_samples,
    )
    from eqvae.losses.vae import beta_for_step  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )
    from eqvae.training.optim import (  # noqa: PLC0415
        SpecAdamWConfig,
        create_adamw_optimizer,
    )
    from eqvae.training.step import TrainStepRequest, run_train_step  # noqa: PLC0415

    row_spec = config.row_spec
    if not torch.cuda.is_available():
        payload = _child_failure_payload(
            settings=config.settings,
            row_spec=row_spec,
            status=WRONG_ACCELERATOR,
            failure_kind="cuda_unavailable",
            failure_message="CUDA is unavailable in child row",
            accelerator=_accelerator_observation(),
        )
        _write_child_payload(config.output_dir, row_spec.row_id, payload)
        return
    accelerator = _accelerator_observation()
    accelerator_failure = _accelerator_failure(
        row_spec=row_spec,
        accelerator=accelerator,
    )
    if accelerator_failure is not None:
        status, failure_kind, failure_message = accelerator_failure
        payload = _child_failure_payload(
            settings=config.settings,
            row_spec=row_spec,
            status=status,
            failure_kind=failure_kind,
            failure_message=failure_message,
            accelerator=accelerator,
        )
        _write_child_payload(config.output_dir, row_spec.row_id, payload)
        return

    device = torch.device("cuda", 0)
    manual_seed = cast("Callable[[int], object]", torch.manual_seed)
    manual_seed(config.settings.global_seed)
    paths = resolve_patch_data_paths(config.data_root)
    train_paths = paths.train
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=train_paths.bin_path,
            csv_path=train_paths.csv_path,
            split=train_paths.split,
            image_size=config.settings.image_size,
            channels=config.settings.channels,
            validate_crc=False,
        ),
    )
    train_indices = _window_indices(config.settings.train_windows)
    subset = Subset(dataset, train_indices)
    loader = cast(
        "DataLoader[PatchTrainingBatch]",
        DataLoader(
            subset,
            batch_size=row_spec.per_device_batch_size,
            shuffle=False,
            num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
            collate_fn=collate_patch_training_samples,
        ),
    )
    raw_model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": config.settings.norm_groups},
    ).to(device)
    # Eps/latent shape follows the built model, never a frozen module constant, so a
    # future model with a different latent width re-runs this pretest unchanged
    # (Spec 0011 R1). Read from the raw model before compile wrapping.
    latent_channels = raw_model.latent_channels
    profile = profile_from_config(config.settings.corruption_config)
    # COMPILE_STEP measures the compiled whole-step recipe single-GPU (Spec 0011 S14b).
    # The fused optimizer + dynamo config come from the S14a-threaded recipe knobs, and
    # inline corruption + forward + FP32 loss fuse into one torch.compile graph -- the
    # single-GPU mirror of the dual-T4 executor branch and the runner S16 compiled step.
    # Every other scope keeps compiled_step_fn None and the byte-identical eager
    # _model_for_compile_scope + grouped-optimizer + _run_one_train_batch path.
    compiled_step_fn: object | None = None
    if row_spec.compile_scope == COMPILE_STEP:
        model, optimizer, compiled_step_fn = _build_compiled_step(
            raw_model=raw_model,
            device=device,
            profile=profile,
            settings=config.settings,
            row_spec=row_spec,
        )
    else:
        model = _model_for_compile_scope(model=raw_model, row_spec=row_spec)
        optimizer, _summary = create_adamw_optimizer(
            raw_model,
            config=SpecAdamWConfig(
                learning_rate=config.settings.learning_rate,
                weight_decay=config.settings.weight_decay,
                gate_lr_multiplier=1.0,
                gradient_clip_global_norm=config.settings.gradient_clip_global_norm,
                beta1=0.9,
                beta2=0.999,
            ),
        )
    iterator = iter(loader)

    def run_one_step(
        step_index: int,
        iterator: Iterator[PatchTrainingBatch],
    ) -> int:
        # A COMPILE_STEP row drives the compiled whole-step closure; every other scope
        # keeps the byte-identical eager train step (``compiled_step_fn`` is None). The
        # loader iterator is passed in (not captured) so the ``finally`` block can still
        # ``del`` it before ``dataset.close()``.
        if compiled_step_fn is not None:
            return _run_compiled_step_batch(
                iterator=iterator,
                compiled_step_fn=compiled_step_fn,
                optimizer=optimizer,
                model=model,
                device=device,
                normalize_uint8_batch_fn=normalize_uint8_batch,
                settings=config.settings,
                step_index=step_index,
                row_spec=row_spec,
                latent_channels=latent_channels,
                beta_for_step_fn=beta_for_step,
            )
        return _run_one_train_batch(
            iterator=iterator,
            model=model,
            optimizer=optimizer,
            device=device,
            profile=profile,
            normalize_uint8_batch_fn=normalize_uint8_batch,
            corrupt_normalized_batch_fn=corrupt_normalized_batch,
            settings=config.settings,
            step_index=step_index,
            row_spec=row_spec,
            latent_channels=latent_channels,
            beta_for_step_fn=beta_for_step,
            train_step_request_factory=TrainStepRequest,
            run_train_step_fn=run_train_step,
        )

    step_ms: list[float] = []
    samples = 0
    compile_startup_sec = 0.0
    post_settle_graph_break_count = 0
    post_settle_recompile_count = 0
    dynamo_counter_source_available = False
    settle_counter_snapshot: JsonObject = {}
    post_settle_counter_snapshot: JsonObject = {}
    try:  # noqa: PLW0717
        if row_spec.compile_scope != COMPILE_NONE:
            dynamo_counter_source_available = _reset_dynamo_counters()
            settle_start_ns = time.perf_counter_ns()
            for step_index in range(config.settings.compile_settle_steps):
                # The whole-step graph must be warmed by the *compiled step* (grad +
                # optimizer), not a forward-only pass: a forward-only settle leaves
                # first-trace compilation for the post-settle window, scoring the row as
                # recompiling (Spec 0011 S14b). run_one_step drives the compiled step
                # for a COMPILE_STEP row and the eager step for every other scope.
                run_one_step(step_index, iterator)
            torch.cuda.synchronize(device)
            compile_startup_sec = _elapsed_seconds(settle_start_ns)
            settle_counter_snapshot = _dynamo_counter_summary()
            _reset_dynamo_counters()
        for step_index in range(config.settings.warmup_steps):
            run_one_step(step_index + config.settings.compile_settle_steps, iterator)
        torch.cuda.reset_peak_memory_stats(device)
        for step_index in range(config.settings.measured_steps):
            start_ns = time.perf_counter_ns()
            batch_size = run_one_step(
                step_index
                + config.settings.compile_settle_steps
                + config.settings.warmup_steps,
                iterator,
            )
            torch.cuda.synchronize(device)
            step_ms.append(_elapsed_ms(start_ns))
            samples += batch_size
        if row_spec.compile_scope != COMPILE_NONE:
            post_settle_counter_snapshot = _dynamo_counter_summary()
            post_settle_graph_break_count = _counter_total(
                post_settle_counter_snapshot,
                "graph_break",
            )
            post_settle_recompile_count = max(
                _counter_total(post_settle_counter_snapshot, "recompil"),
                _counter_total(post_settle_counter_snapshot, "unique_graphs"),
            )
    except (RuntimeError, StopIteration, ValueError) as exc:
        payload = _child_failure_payload(
            settings=config.settings,
            row_spec=row_spec,
            status=RUNTIME_ERROR,
            failure_kind=f"runtime_{type(exc).__name__}",
            failure_message=str(exc),
            accelerator=accelerator,
        )
        _write_child_payload(config.output_dir, row_spec.row_id, payload)
        return
    finally:
        del iterator
        del loader
        dataset.close()

    payload = cast(
        "JsonObject",
        {
            "row_id": row_spec.row_id,
            "status": PASS_STATUS,
            "accelerator": accelerator,
            "step_ms": step_ms,
            "samples": samples,
            "max_vram_allocated_mb": _cuda_allocated_mb(device),
            "max_vram_reserved_mb": _cuda_reserved_mb(device),
            "vram_headroom_fraction": _cuda_headroom_fraction(device),
            "compile_startup_sec": compile_startup_sec,
            "dynamo_counter_source_available": dynamo_counter_source_available,
            "settle_counter_snapshot": settle_counter_snapshot,
            "post_settle_counter_snapshot": post_settle_counter_snapshot,
            "post_settle_graph_break_count": post_settle_graph_break_count,
            "post_settle_recompile_count": post_settle_recompile_count,
        },
    )
    _write_child_payload(config.output_dir, row_spec.row_id, payload)


def _run_one_train_batch(  # noqa: PLR0913
    *,
    iterator: Iterator[PatchTrainingBatch],
    model: NonEquivariantVAE,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    profile: StainCorruptionProfile,
    normalize_uint8_batch_fn: NormalizeUint8BatchFn,
    corrupt_normalized_batch_fn: CorruptNormalizedBatchFn,
    settings: RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: RowSpec,
    latent_channels: int,
    beta_for_step_fn: BetaForStepFn,
    train_step_request_factory: TrainStepRequestFactory,
    run_train_step_fn: TrainStepRunner,
) -> int:
    import torch  # noqa: PLC0415

    batch = next(iterator)
    clean = normalize_uint8_batch_fn(batch.images_uint8).to(device=device)
    corruption = corrupt_normalized_batch_fn(
        clean,
        profile=profile,
        corruption_seed=settings.corruption_seed,
        split=batch.split,
        semantic_sample_keys=batch.semantic_sample_keys,
        corruption_step=step_index,
        corruption_view="train_corrupted_real_data_runtime_pretest",
        strategy=row_spec.corruption_strategy,
    )
    shape = cast("tuple[int, int, int, int]", tuple(clean.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    beta = beta_for_step_fn(
        optimizer_step_index=step_index,
        max_optimizer_steps=settings.warmup_steps + settings.measured_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    request = train_step_request_factory(
        model=model,
        optimizer=optimizer,
        clean_batch=clean,
        eps=eps,
        beta=beta,
        ssim_weight=settings.ssim_weight,
        optimizer_step_index=step_index,
        gradient_clip_global_norm=settings.gradient_clip_global_norm,
        input_batch=corruption.corrupted,
    )
    run_train_step_fn(request)
    return shape[0]


# The compiled whole-step backend. Matches the generator's derived plan value
# (`runtime_selection._selected_runtime_payload`: any compiled scope -> "inductor") and
# the dual-T4 executor's ``_STEP_COMPILE_BACKEND``, so the single-GPU stability screen
# compiles the step under the same backend the dual-T4 measurement and the runner use.
_STEP_COMPILE_BACKEND = "inductor"


def _build_compiled_step(
    *,
    raw_model: NonEquivariantVAE,
    device: torch.device,
    profile: StainCorruptionProfile,
    settings: RealDataRuntimePretestSettings,
    row_spec: RowSpec,
) -> tuple[NonEquivariantVAE, torch.optim.AdamW, object]:
    """Build the fused optimizer and compiled whole-step closure for a step row.

    Single-GPU mirror of the dual-T4 executor's ``_build_compiled_ddp_step`` (and the
    runner's ``_maybe_build_compiled_step``, Spec 0011 S16): the fused AdamW and the
    process-global dynamo config come from the S14a-threaded recipe knobs, and inline
    corruption + the fp32 forward + the FP32 loss island fuse into one
    ``torch.compile(dynamic=False)`` graph, so the single-GPU pre-screen exercises the
    same compiled step the dual-T4 executor measures and the runner consumes. There is
    no ``DistributedDataParallel`` wrap (this path is single-GPU), so the model is
    returned unwrapped and the ``ddp_static_graph`` interleaving concern does not arise:
    this child measures timing only; the eager numerical proof lives in
    ``_one_strategy_train_step_evidence``.

    The one fail-closed precondition mirrors the executor: ``amp_off_fp32`` only, since
    the closure hardcodes ``autocast_enabled=False`` and no GradScaler.
    ``_run_stage1_rows`` already screens non-fp32 rows, so the guard is defense-in-depth
    and an honest failure if that contract ever changes.

    Returns:
        A ``(model, optimizer, compiled_step_fn)`` triple; ``model`` is the raw model
        the compiled closure and the optimizer share.

    Raises:
        ValueError: If the step row requests an AMP precision policy the compiled-step
            screen cannot faithfully mirror to the runner.

    """
    import torch  # noqa: PLC0415

    from eqvae.corruption.inline_stain import InlineStainCorruptor  # noqa: PLC0415
    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        apply_fastpath_dynamo_config,
        build_fastpath_optimizer,
    )
    from eqvae.training.fastpath_step import make_fastpath_step_fn  # noqa: PLC0415
    from eqvae.training.optim import SpecAdamWConfig  # noqa: PLC0415

    if row_spec.precision_policy != AMP_OFF_FP32:
        message = (
            "Compiled whole-step measurement mirrors the runner's fp32 fast path only "
            "(autocast and GradScaler are not wired into the compiled-step screen), "
            f"so precision_policy must be {AMP_OFF_FP32!r}; got "
            f"{row_spec.precision_policy!r}."
        )
        raise ValueError(message)
    optimizer = build_fastpath_optimizer(
        raw_model,
        config=SpecAdamWConfig(
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            gate_lr_multiplier=1.0,
            gradient_clip_global_norm=settings.gradient_clip_global_norm,
            beta1=0.9,
            beta2=0.999,
            fused=row_spec.fused_optimizer,
        ),
    )
    apply_fastpath_dynamo_config(
        optimize_ddp=row_spec.optimize_ddp,
        compiled_autograd=row_spec.compiled_autograd,
        reorder_compute_comm_overlap=row_spec.reorder_compute_comm_overlap,
    )
    corruptor = InlineStainCorruptor(profile).to(device=device)
    step_fn = make_fastpath_step_fn(
        raw_model,
        corruptor,
        ssim_weight=settings.ssim_weight,
        autocast_dtype=torch.float32,
        autocast_enabled=False,
    )
    compiled_step_fn = cast("Callable[..., object]", torch.compile)(
        step_fn,
        dynamic=False,
        backend=_STEP_COMPILE_BACKEND,
    )
    return raw_model, optimizer, compiled_step_fn


def _run_compiled_step_batch(  # noqa: PLR0913
    *,
    iterator: Iterator[PatchTrainingBatch],
    compiled_step_fn: object,
    optimizer: torch.optim.Optimizer,
    model: NonEquivariantVAE,
    device: torch.device,
    normalize_uint8_batch_fn: NormalizeUint8BatchFn,
    settings: RealDataRuntimePretestSettings,
    step_index: int,
    row_spec: RowSpec,
    latent_channels: int,
    beta_for_step_fn: BetaForStepFn,
) -> int:
    """Drive one compiled whole-step batch (settle / warmup / measured), single-GPU.

    Single-GPU mirror of the executor's ``_run_compiled_ddp_step_batch``: the compiled
    closure fuses inline corruption + the forward + the FP32 loss; the backward, grad
    clipping, and the optimizer step stay eager here, exactly as the runner drives the
    recipe. Only the clean batch, ``eps``, and a 0-dim ``beta`` tensor cross the graph
    boundary. Step rows are ``amp_off_fp32`` (no GradScaler), so no AMP-skip exists.
    Pretest rows are always ``memory_format="contiguous"`` (``_stage1_row_specs`` never
    sets channels_last), so there is no layout conversion here (unlike the executor,
    whose rows can be channels_last).

    Returns:
        The observed batch size, matching ``_run_one_train_batch`` so the measured loop
        accumulates ``samples`` identically.

    """
    import torch  # noqa: PLC0415

    from eqvae.training.fastpath_recipe import (  # noqa: PLC0415
        compiled_autograd_context,
    )

    batch = next(iterator)
    clean = normalize_uint8_batch_fn(batch.images_uint8).to(device=device)
    shape = cast("tuple[int, int, int, int]", tuple(clean.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    beta_value = beta_for_step_fn(
        optimizer_step_index=step_index,
        max_optimizer_steps=settings.warmup_steps + settings.measured_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    # beta crosses the compiled graph boundary as a 0-dim tensor so the warmup schedule
    # changing its value never forces a ``dynamic=False`` recompile.
    beta = torch.tensor(float(beta_value), dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=row_spec.zero_grad_set_to_none)
    with compiled_autograd_context(enabled=row_spec.compiled_autograd):
        output = cast(
            "FastpathStepOutput",
            cast("Callable[..., object]", compiled_step_fn)(clean, eps, beta),
        )
        backward = cast("Callable[[], None]", output.loss.backward)
        backward()
    if settings.gradient_clip_global_norm > 0.0:
        cast("Callable[..., object]", torch.nn.utils.clip_grad_norm_)(
            list(model.parameters()),
            settings.gradient_clip_global_norm,
            foreach=row_spec.gradient_clip_foreach,
        )
    optimizer.step()
    return shape[0]


def _model_for_compile_scope(
    *,
    model: NonEquivariantVAE,
    row_spec: RowSpec,
) -> NonEquivariantVAE:
    return _model_for_compile_scope_name(
        model=model,
        compile_scope=row_spec.compile_scope,
    )


def _model_for_compile_scope_name(
    *,
    model: NonEquivariantVAE,
    compile_scope: str,
) -> NonEquivariantVAE:
    # COMPILE_STEP compiles the whole train-step closure (see ``_build_compiled_step``),
    # not the model object, so the model is returned unwrapped here and invoked eagerly.
    # This keeps the paired numerical proof (``_one_strategy_train_step_evidence``) on
    # the eager path for a step row -- the compiled ``FastpathStepOutput`` cannot emit
    # the mu/logvar/corruption-hash/gate telemetry that lane records -- mirroring the
    # dual-T4 executor's ``_compile_ddp_model_if_requested``.
    if compile_scope in {COMPILE_NONE, COMPILE_STEP}:
        return model
    if compile_scope == COMPILE_MODEL_FORWARD:
        compile_fn = cast("Callable[..., object]", torch.compile)
        return cast("NonEquivariantVAE", compile_fn(model))
    message = f"Unsupported compile scope in child row: {compile_scope}"
    raise ValueError(message)


def _settings(
    resolved: ResolvedConfig,
    *,
    data_root_override: str | None,
) -> RealDataRuntimePretestSettings:
    effective = resolved.effective_config
    data = _required_object(effective, "data")
    runtime = _required_object(effective, "runtime_matrix")
    pretest = _required_object(effective, "runtime_pretest")
    benchmark_cap = _required_object(data, "benchmark_cap")
    objective = _required_object(effective, "objective")
    beta = _required_object(objective, "beta")
    optimizer = _required_object(effective, "optimizer")
    seeds = _required_object(effective, "seeds")
    model = _required_object(effective, "model")
    normalization = _required_object(model, "normalization")
    _validate_pretest_contract(pretest)
    return RealDataRuntimePretestSettings(
        run_name=_required_str(_required_object(effective, "run"), "name"),
        dataset_slug=_required_str(data, "dataset_slug"),
        data_root=data_root_override or _required_str(data, "data_root"),
        image_size=_optional_int(data, "image_size") or _image_size_from_model(model),
        channels=_optional_int(data, "channels") or _channels_from_model(model),
        real_train_patch_count=_required_int(data, "real_train_patch_count"),
        real_validation_patch_count=_required_int(data, "real_validation_patch_count"),
        cap_train_patch_count=_required_int(benchmark_cap, "train_patch_count"),
        cap_validation_patch_count=_required_int(
            benchmark_cap,
            "validation_patch_count",
        ),
        window_policy=_required_str(benchmark_cap, "window_policy"),
        train_windows=_window_specs(benchmark_cap, "train_windows"),
        validation_windows=_window_specs(benchmark_cap, "validation_windows"),
        warmup_steps=_required_int(runtime, "warmup_steps"),
        measured_steps=_required_int(runtime, "measured_steps"),
        repeats=_required_int(runtime, "repeats"),
        compile_settle_steps=_required_int(
            _required_object(runtime, "compile_settle_policy"),
            "compile_settle_steps",
        ),
        compile_scopes=_str_tuple(runtime, "compile_scopes"),
        corruption_strategies=_str_tuple(runtime, "corruption_strategies"),
        seeded_candidates=_seeded_candidates(runtime),
        blocked_claims=_required_object(pretest, "blocked_claims"),
        ssim_weight=_required_float(objective, "ssim_weight"),
        beta_target=_required_float(beta, "target"),
        beta_warmup_fraction=_required_float(beta, "step_limited_warmup_fraction"),
        learning_rate=_required_float(optimizer, "learning_rate"),
        weight_decay=_required_float(optimizer, "weight_decay"),
        gradient_clip_global_norm=_required_float(
            optimizer,
            "gradient_clip_global_norm",
        ),
        global_seed=_required_int(seeds, "global_seed"),
        data_seed=_required_int(seeds, "data_seed"),
        corruption_seed=_required_int(seeds, "corruption_seed"),
        latent_seed=_required_int(seeds, "latent_seed"),
        corruption_config=_required_object(effective, "corruption"),
        norm_groups=_required_int(normalization, "num_groups"),
    )


def _validate_pretest_contract(pretest: JsonObject) -> None:
    expected = {
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "status_scope": REAL_DATA_PRETEST_SCOPE,
    }
    for key, value in expected.items():
        if _required_str(pretest, key) != value:
            message = f"runtime_pretest.{key} must be {value!r}"
            raise ValueError(message)
    if _required_bool(pretest, "full_run_eligible"):
        message = "runtime_pretest.full_run_eligible must be false"
        raise ValueError(message)
    if _required_bool(pretest, "writes_selected_runtime"):
        message = "runtime_pretest.writes_selected_runtime must be false"
        raise ValueError(message)


def _stage1_row_specs(settings: RealDataRuntimePretestSettings) -> tuple[RowSpec, ...]:
    specs: list[RowSpec] = []
    for candidate in settings.seeded_candidates:
        world_size, nproc_per_node, cuda_visible_devices = _row_runtime(
            candidate.accelerator_mode,
        )
        for compile_scope in settings.compile_scopes:
            specs.extend(
                RowSpec(
                    row_id=_row_id(
                        accelerator_mode=candidate.accelerator_mode,
                        per_device_batch_size=candidate.per_device_batch_size,
                        precision_policy=AMP_OFF_FP32,
                        compile_scope=compile_scope,
                        corruption_strategy=corruption_strategy,
                    ),
                    accelerator_mode=candidate.accelerator_mode,
                    per_device_batch_size=candidate.per_device_batch_size,
                    precision_policy=AMP_OFF_FP32,
                    compile_scope=compile_scope,
                    corruption_strategy=corruption_strategy,
                    parent_synthetic_row_id=candidate.synthetic_v4_row_id or "",
                    candidate_role=candidate.candidate_role,
                    world_size=world_size,
                    nproc_per_node=nproc_per_node,
                    cuda_visible_devices=cuda_visible_devices,
                )
                for corruption_strategy in settings.corruption_strategies
            )
    return tuple(specs)


def _row_runtime(accelerator_mode: str) -> tuple[int, int, str]:
    if accelerator_mode == SINGLE_VISIBLE_T4:
        return (1, 1, "0")
    if accelerator_mode == DUAL_T4_DDP:
        return (DUAL_T4_DEVICE_COUNT, DUAL_T4_DEVICE_COUNT, "0,1")
    message = f"Unsupported accelerator_mode: {accelerator_mode}"
    raise ValueError(message)


def _row_id(
    *,
    accelerator_mode: str,
    per_device_batch_size: int,
    precision_policy: str,
    compile_scope: str,
    corruption_strategy: str,
) -> str:
    return (
        f"{accelerator_mode}__bs{per_device_batch_size}__{precision_policy}"
        f"__compile_{compile_scope}__{corruption_strategy}"
    )


def _row_from_child_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    payload: JsonObject,
) -> CsvRow:
    row_spec = _row_spec_from_id(settings, _required_str(payload, "row_id"))
    status = _required_str(payload, "status")
    if status != PASS_STATUS:
        return _failure_row(
            settings=settings,
            row_spec=row_spec,
            status=status,
            failure_kind=_required_str(payload, "failure_kind"),
            failure_message=_required_str(payload, "failure_message"),
            accelerator=_required_object(payload, "accelerator"),
        )
    step_ms = _float_list(payload, "step_ms")
    steady_p50 = _percentile(step_ms, 0.50)
    global_batch_size = row_spec.per_device_batch_size * row_spec.world_size
    samples_sec = (
        0.0 if steady_p50 <= 0.0 else global_batch_size / (steady_p50 / 1000.0)
    )
    row = dict(_base_row(settings=settings, row_spec=row_spec))
    accelerator = _required_object(payload, "accelerator")
    row.update({
        "visible_device_count": str(_required_int(accelerator, "visible_device_count")),
        "cuda_device_count": str(_required_int(accelerator, "cuda_device_count")),
        "gpu_names": json.dumps(_required_str_list(accelerator, "gpu_names")),
        "compile_startup_sec": _format_float(
            _optional_float(payload, "compile_startup_sec") or 0.0,
        ),
        "steady_step_ms_p50": _format_float(steady_p50),
        "steady_step_ms_p95": _format_float(_percentile(step_ms, 0.95)),
        "samples_sec": _format_float(samples_sec),
        "trainer_samples_sec": _format_float(samples_sec),
        "max_vram_allocated_mb": _format_float(
            _required_float(payload, "max_vram_allocated_mb"),
        ),
        "max_vram_reserved_mb": _format_float(
            _required_float(payload, "max_vram_reserved_mb"),
        ),
        "vram_headroom_fraction": _format_float(
            _required_float(payload, "vram_headroom_fraction"),
        ),
        "status": INELIGIBLE_STATUS,
        "failure_kind": "linked_safety_evidence_pending",
        "failure_message_hash": _hash_text("linked_safety_evidence_pending"),
    })
    if row_spec.compile_scope != COMPILE_NONE and _required_bool(
        payload,
        "dynamo_counter_source_available",
    ):
        row["graph_break_count"] = str(
            _optional_int(payload, "post_settle_graph_break_count") or 0,
        )
        row["recompile_count"] = str(
            _optional_int(payload, "post_settle_recompile_count") or 0,
        )
    return row


def _row_spec_from_id(settings: RealDataRuntimePretestSettings, row_id: str) -> RowSpec:
    for row_spec in _stage1_row_specs(settings):
        if row_spec.row_id == row_id:
            return row_spec
    message = f"Unknown child row_id: {row_id}"
    raise ValueError(message)


def _unsupported_row(
    *,
    settings: RealDataRuntimePretestSettings,
    row_spec: RowSpec,
    failure_kind: str,
) -> CsvRow:
    return _failure_row(
        settings=settings,
        row_spec=row_spec,
        status=SKIPPED_UNSUPPORTED,
        failure_kind=failure_kind,
        failure_message=failure_kind,
    )


def _failure_row(  # noqa: PLR0913
    *,
    settings: RealDataRuntimePretestSettings,
    row_spec: RowSpec,
    status: str,
    failure_kind: str,
    failure_message: str,
    accelerator: JsonObject | None = None,
) -> CsvRow:
    row = dict(_base_row(settings=settings, row_spec=row_spec))
    observed = accelerator or {
        "visible_device_count": 0,
        "cuda_device_count": 0,
        "gpu_names": [],
    }
    row.update({
        "visible_device_count": str(_required_int(observed, "visible_device_count")),
        "cuda_device_count": str(_required_int(observed, "cuda_device_count")),
        "gpu_names": json.dumps(_required_str_list(observed, "gpu_names")),
        "status": status,
        "failure_kind": failure_kind,
        "failure_message_hash": _hash_text(failure_message),
    })
    return row


def _base_row(*, settings: RealDataRuntimePretestSettings, row_spec: RowSpec) -> CsvRow:
    return {
        "run_name": settings.run_name,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "row_id": row_spec.row_id,
        "accelerator_mode": row_spec.accelerator_mode,
        "machine_shape": "NvidiaTeslaT4",
        "visible_device_count": "",
        "cuda_device_count": "",
        "gpu_names": "[]",
        "ddp_backend": "nccl" if row_spec.accelerator_mode == DUAL_T4_DDP else "",
        "world_size": str(row_spec.world_size),
        "nproc_per_node": str(row_spec.nproc_per_node),
        "precision_policy": row_spec.precision_policy,
        "amp_enabled": "false",
        "torch_compile_enabled": _format_bool(
            value=row_spec.compile_scope != COMPILE_NONE,
        ),
        "compile_scope": row_spec.compile_scope,
        "runtime_policy_id": row_spec.runtime_policy_id,
        "memory_format": row_spec.memory_format,
        "autocast_dtype": row_spec.autocast_dtype,
        "fp32_loss": _format_bool(value=row_spec.fp32_loss),
        "grad_scaler_enabled": _format_bool(value=row_spec.grad_scaler_enabled),
        "cudnn_benchmark": _format_bool(value=row_spec.cudnn_benchmark),
        "cudnn_deterministic": _format_bool(value=row_spec.cudnn_deterministic),
        "deterministic_algorithms": _format_bool(
            value=row_spec.deterministic_algorithms,
        ),
        "tf32_enabled": _format_bool(value=row_spec.tf32_enabled),
        "matmul_precision": row_spec.matmul_precision,
        "ddp_static_graph": _format_bool(value=row_spec.ddp_static_graph),
        "ddp_gradient_as_bucket_view": _format_bool(
            value=row_spec.ddp_gradient_as_bucket_view,
        ),
        "optimizer_implementation": row_spec.optimizer_implementation,
        "zero_grad_set_to_none": _format_bool(value=row_spec.zero_grad_set_to_none),
        "gradient_clip_foreach": _format_bool(value=row_spec.gradient_clip_foreach),
        "compile_dynamic": _format_bool(value=row_spec.compile_dynamic),
        # Spec 0011 S14a: recipe knobs from the row via the shared producer helper.
        **_recipe_knob_columns(row_spec=row_spec),
        "corruption_strategy": row_spec.corruption_strategy,
        "per_device_batch_size": str(row_spec.per_device_batch_size),
        "global_batch_size": str(row_spec.per_device_batch_size * row_spec.world_size),
        "gradient_accumulation_steps": "1",
        "warmup_steps": str(settings.warmup_steps),
        "measured_steps": str(settings.measured_steps),
        "repeats": str(settings.repeats),
        "compile_startup_sec": "0.000000",
        "compile_settle_steps": (
            "0"
            if row_spec.compile_scope == COMPILE_NONE
            else str(settings.compile_settle_steps)
        ),
        "steady_step_ms_p50": "",
        "steady_step_ms_p95": "",
        "samples_sec": "",
        "trainer_samples_sec": "",
        "max_vram_allocated_mb": "",
        "max_vram_reserved_mb": "",
        "vram_headroom_fraction": "",
        "amp_step_skipped_count": "",
        "gate_health_status": "skipped_unsupported",
        "gate_health_warning_count": "",
        "numerical_check_status": "skipped_unsupported",
        "data_wait_fraction_p95": "",
        "graph_break_count": "",
        "recompile_count": "",
        "oom": "false",
        "status": "",
        "failure_kind": "",
        "failure_message_hash": "",
    }


def _real_data_identity_and_clean_path_proof(
    settings: RealDataRuntimePretestSettings,
) -> JsonObject:
    try:
        return _real_data_identity_and_clean_path_proof_or_raise(settings)
    except DataRootUnavailableError as exc:
        return _data_proof_failure_payload(
            settings=settings,
            status=SKIPPED_UNSUPPORTED,
            failure_kind="data_root_unavailable",
            failure_message=str(exc),
            data_root_diagnostics=exc.diagnostics,
        )
    except FileNotFoundError as exc:
        return _data_proof_failure_payload(
            settings=settings,
            status="fail",
            failure_kind="data_proof_FileNotFoundError",
            failure_message=str(exc),
            data_root_diagnostics=_data_root_diagnostics(settings.data_root),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _data_proof_failure_payload(
            settings=settings,
            status="fail",
            failure_kind=f"data_proof_{type(exc).__name__}",
            failure_message=str(exc),
            data_root_diagnostics=_data_root_diagnostics(settings.data_root),
        )


def _real_data_identity_and_clean_path_proof_or_raise(
    settings: RealDataRuntimePretestSettings,
) -> JsonObject:
    paths = _resolve_patch_data_paths_for_pretest(settings.data_root)
    train_proof = _split_identity_proof(
        split_paths=paths.train,
        settings=settings,
        windows=settings.train_windows,
    )
    validation_proof = _split_identity_proof(
        split_paths=paths.validation,
        settings=settings,
        windows=settings.validation_windows,
    )
    if _split_windows_pass(train_proof) and _split_windows_pass(validation_proof):
        clean_validation_proof = _clean_validation_dataloader_proof(
            paths=paths,
            settings=settings,
        )
    else:
        clean_validation_proof = _clean_validation_not_run_payload(
            failure_kind="window_proof_failed",
        )
    return _data_proof_payload(
        settings=settings,
        paths=paths,
        train_proof=train_proof,
        validation_proof=validation_proof,
        clean_validation_proof=clean_validation_proof,
    )


def _split_identity_proof(
    *,
    split_paths: PatchSplitPaths,
    settings: RealDataRuntimePretestSettings,
    windows: Sequence[WindowSpec],
) -> JsonObject:
    from eqvae.data.patch_shards import (  # noqa: PLC0415
        PATCH_SHARD_HEADER_SIZE,
        PatchShard,
        PatchShardSpec,
    )

    shard = PatchShard(
        PatchShardSpec(
            bin_path=split_paths.bin_path,
            csv_path=split_paths.csv_path,
            image_size=settings.image_size,
            channels=settings.channels,
            validate_crc=False,
        ),
    )
    binary_integrity = _binary_integrity_payload(
        bin_path=split_paths.bin_path,
        header_size=PATCH_SHARD_HEADER_SIZE,
        expected_crc32=shard.header.crc32,
    )
    csv_sha256 = _sha256_file(split_paths.csv_path)
    expected_count = _expected_patch_count(settings=settings, split=split_paths.split)
    records = shard.records
    window_proof = _windows_proof(
        split=split_paths.split,
        records=records,
        windows=windows,
    )
    row_count_pass = len(records) == expected_count and shard.header.patch_count == len(
        records,
    )
    crc_pass = _required_bool(binary_integrity, "crc_matches_header")
    window_pass = _required_str(window_proof, "status") == PASS_STATUS
    status = PASS_STATUS if row_count_pass and crc_pass and window_pass else "fail"
    label_counts = Counter(record.label for record in records)
    return cast(
        "JsonObject",
        {
            "status": status,
            "split": split_paths.split,
            "bin_path": str(split_paths.bin_path),
            "csv_path": str(split_paths.csv_path),
            "file_hashes": [
                {
                    "split": split_paths.split,
                    "kind": "binary",
                    "path": str(split_paths.bin_path),
                    "sha256": _required_str(binary_integrity, "sha256"),
                    "size_bytes": _required_int(binary_integrity, "size_bytes"),
                },
                {
                    "split": split_paths.split,
                    "kind": "csv",
                    "path": str(split_paths.csv_path),
                    "sha256": csv_sha256,
                    "size_bytes": split_paths.csv_path.stat().st_size,
                },
            ],
            "header": {
                "crc32": shard.header.crc32,
                "patch_count": shard.header.patch_count,
                "channels": shard.header.channels,
                "height": shard.header.height,
                "width": shard.header.width,
                "version": shard.header.version,
                "layout": shard.header.layout.decode("ascii"),
            },
            "observed_crc32": _required_int(binary_integrity, "observed_crc32"),
            "crc_matches_header": crc_pass,
            "crc_validation_status": PASS_STATUS if crc_pass else "fail",
            "csv_row_count": len(records),
            "expected_patch_count": expected_count,
            "row_count_matches_config": row_count_pass,
            "row_count_status": PASS_STATUS if row_count_pass else "fail",
            "unique_wsi_count": len({record.wsi_id for record in records}),
            "wsi_ids": sorted({record.wsi_id for record in records}),
            "wsi_count_status": PASS_STATUS if records else "fail",
            "label_counts": {
                str(label): label_counts[label] for label in sorted(label_counts)
            },
            "first_record": _record_identity_payload(
                records[0],
                split=split_paths.split,
            ),
            "last_record": _record_identity_payload(
                records[-1],
                split=split_paths.split,
            ),
            "windows": window_proof,
        },
    )


def _data_proof_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    paths: PatchDataPaths,
    train_proof: JsonObject,
    validation_proof: JsonObject,
    clean_validation_proof: JsonObject,
) -> JsonObject:
    train_status = _required_str(train_proof, "status")
    validation_status = _required_str(validation_proof, "status")
    clean_status = _required_str(clean_validation_proof, "status")
    split_contract = _split_contract_proof(
        settings=settings,
        train_proof=train_proof,
        validation_proof=validation_proof,
    )
    window_contract = _window_contract_proof(settings)
    technical_identity_pass = (
        train_status == PASS_STATUS
        and validation_status == PASS_STATUS
        and _required_str(split_contract, "status") != "fail"
        and _required_str(window_contract, "status") != "fail"
    )
    identity_status = _identity_status(
        settings=settings,
        technical_identity_pass=technical_identity_pass,
        split_contract=split_contract,
        window_contract=window_contract,
    )
    overall_status = (
        PASS_STATUS
        if identity_status == PASS_STATUS and clean_status == PASS_STATUS
        else identity_status
        if identity_status == "local_pass" and clean_status == PASS_STATUS
        else "fail"
    )
    return {
        "status": overall_status,
        "identity_status": identity_status,
        "row_count_status": _combined_status(
            _required_str(train_proof, "row_count_status"),
            _required_str(validation_proof, "row_count_status"),
        ),
        "wsi_count_status": _required_str(split_contract, "status"),
        "crc_validation_status": _combined_status(
            _required_str(train_proof, "crc_validation_status"),
            _required_str(validation_proof, "crc_validation_status"),
        ),
        "window_status": _required_str(window_contract, "status"),
        "clean_validation_dataloader_status": clean_status,
        "dataset_slug": settings.dataset_slug,
        "data_root": settings.data_root,
        "resolved_data_root": str(paths.root),
        "data_root_diagnostics": _data_root_diagnostics(settings.data_root),
        "canonical_real_contract": _canonical_real_contract(settings),
        "split_contract": split_contract,
        "window_contract": window_contract,
        "file_hashes": [
            *_required_object_list(train_proof, "file_hashes"),
            *_required_object_list(validation_proof, "file_hashes"),
        ],
        "splits": {
            "train": train_proof,
            "validation": validation_proof,
        },
        "clean_validation_dataloader": clean_validation_proof,
    }


def _data_proof_failure_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    status: str,
    failure_kind: str,
    failure_message: str,
    data_root_diagnostics: JsonObject,
) -> JsonObject:
    return {
        "status": status,
        "identity_status": status,
        "row_count_status": status,
        "wsi_count_status": status,
        "crc_validation_status": status,
        "window_status": status,
        "clean_validation_dataloader_status": status,
        "dataset_slug": settings.dataset_slug,
        "data_root": settings.data_root,
        "resolved_data_root": "",
        "data_root_diagnostics": data_root_diagnostics,
        "file_hashes": [],
        "splits": {},
        "clean_validation_dataloader": {
            "status": status,
            "failure_kind": failure_kind,
        },
        "failure_kind": failure_kind,
        "failure_message_excerpt": _failure_message_excerpt(failure_message),
        "failure_message_hash": _hash_text(failure_message),
    }


def _resolve_patch_data_paths_for_pretest(data_root: str) -> PatchDataPaths:
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415

    attempts = (
        DATA_ROOT_RESOLUTION_ATTEMPTS if _should_retry_data_root(data_root) else 1
    )
    last_message = ""
    diagnostics = _data_root_diagnostics(data_root)
    for attempt in range(1, attempts + 1):
        diagnostics = _data_root_diagnostics(data_root)
        _print_data_root_diagnostics(
            event="data_root_probe",
            attempt=attempt,
            attempts=attempts,
            diagnostics=diagnostics,
        )
        try:
            return resolve_patch_data_paths(data_root)
        except FileNotFoundError as exc:
            last_message = str(exc)
            if attempt >= attempts:
                break
            time.sleep(DATA_ROOT_RETRY_SLEEP_SEC)
    raise DataRootUnavailableError(last_message, diagnostics=diagnostics)


def _should_retry_data_root(data_root: str) -> bool:
    return data_root == "auto" and Path("/kaggle/input").exists()


def _data_root_diagnostics(data_root: str) -> JsonObject:
    from eqvae.data.roots import data_root_resolution_diagnostics  # noqa: PLC0415

    return cast("JsonObject", data_root_resolution_diagnostics(data_root))


def _print_data_root_diagnostics(
    *,
    event: str,
    attempt: int,
    attempts: int,
    diagnostics: JsonObject,
) -> None:
    if not _diagnostic_bool(diagnostics, "kaggle_input_exists"):
        return
    summary: JsonObject = {
        "requested_data_root": _diagnostic_str(diagnostics, "requested_data_root"),
        "kaggle_input_exists": True,
        "kaggle_input_scan_truncated": _diagnostic_bool(
            diagnostics,
            "kaggle_input_scan_truncated",
        ),
        "candidate_count": _diagnostic_int(diagnostics, "candidate_count"),
        "accepted_candidate_count": _diagnostic_int(
            diagnostics,
            "accepted_candidate_count",
        ),
        "complete_unaccepted_candidate_count": _diagnostic_int(
            diagnostics,
            "complete_unaccepted_candidate_count",
        ),
        "accepted_candidate_roots": _diagnostic_candidate_roots(
            diagnostics,
            "accepted_candidates",
        ),
        "complete_unaccepted_candidate_roots": _diagnostic_candidate_roots(
            diagnostics,
            "complete_unaccepted_candidates",
        ),
        "snapshot_entry_count": len(
            _diagnostic_list(diagnostics, "kaggle_input_snapshot"),
        ),
    }
    payload = {
        "event": f"real_data_runtime_pretest_{event}",
        "attempt": attempt,
        "attempts": attempts,
        "diagnostics": summary,
    }
    sys.stderr.write(f"{json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def _diagnostic_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    return value if isinstance(value, bool) else False


def _diagnostic_int(payload: JsonObject, key: str) -> int:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _diagnostic_str(payload: JsonObject, key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _diagnostic_list(payload: JsonObject, key: str) -> list[JsonValue]:
    value = payload.get(key)
    return value if isinstance(value, list) else []


def _diagnostic_candidate_roots(payload: JsonObject, key: str) -> list[JsonValue]:
    roots: list[JsonValue] = []
    for item in _diagnostic_list(payload, key):
        if not isinstance(item, dict):
            continue
        candidate_root = item.get("candidate_root")
        if isinstance(candidate_root, str):
            roots.append(candidate_root)
    return roots


def _failure_message_excerpt(message: str) -> str:
    single_line = " ".join(message.split())
    return single_line[:256]


def _split_windows_pass(split_proof: JsonObject) -> bool:
    return (
        _required_str(_required_object(split_proof, "windows"), "status") == PASS_STATUS
    )


def _clean_validation_not_run_payload(*, failure_kind: str) -> JsonObject:
    return {
        "status": "fail",
        "split": "validation",
        "proof_scope": "validation_loader_clean_input_only",
        "failure_kind": failure_kind,
    }


def _identity_status(
    *,
    settings: RealDataRuntimePretestSettings,
    technical_identity_pass: bool,
    split_contract: JsonObject,
    window_contract: JsonObject,
) -> str:
    if not technical_identity_pass:
        return "fail"
    if (
        _canonical_real_contract(settings)
        and _required_str(split_contract, "status") == PASS_STATUS
        and _required_str(window_contract, "status") == PASS_STATUS
    ):
        return PASS_STATUS
    return "local_pass"


def _split_contract_proof(
    *,
    settings: RealDataRuntimePretestSettings,
    train_proof: JsonObject,
    validation_proof: JsonObject,
) -> JsonObject:
    from eqvae.data.splits import load_masked_holdout_wsi_ids  # noqa: PLC0415

    train_wsi_ids = set(_required_str_list(train_proof, "wsi_ids"))
    validation_wsi_ids = set(_required_str_list(validation_proof, "wsi_ids"))
    holdout_wsi_ids = set(
        load_masked_holdout_wsi_ids(
            _repo_root() / "docs/data/ubc_ocean_masked_holdout_ids.csv",
        ),
    )
    overlap = tuple(sorted(train_wsi_ids.intersection(validation_wsi_ids)))
    masked_overlap = tuple(
        sorted(train_wsi_ids.union(validation_wsi_ids).intersection(holdout_wsi_ids)),
    )
    canonical = _canonical_real_counts_and_source(settings)
    train_wsi_count_matches = len(train_wsi_ids) == EXPECTED_REAL_TRAIN_WSI_COUNT
    validation_wsi_count_matches = (
        len(validation_wsi_ids) == EXPECTED_REAL_VALIDATION_WSI_COUNT
    )
    status = _real_or_local_contract_status(
        canonical=canonical,
        real_pass=(
            train_wsi_count_matches
            and validation_wsi_count_matches
            and not overlap
            and not masked_overlap
        ),
        local_pass=not overlap and not masked_overlap,
    )
    return cast(
        "JsonObject",
        {
            "status": status,
            "mode": "real" if canonical else "local",
            "expected_train_wsi_count": (
                EXPECTED_REAL_TRAIN_WSI_COUNT if canonical else None
            ),
            "expected_validation_wsi_count": (
                EXPECTED_REAL_VALIDATION_WSI_COUNT if canonical else None
            ),
            "train_wsi_count": len(train_wsi_ids),
            "validation_wsi_count": len(validation_wsi_ids),
            "train_wsi_count_matches": train_wsi_count_matches,
            "validation_wsi_count_matches": validation_wsi_count_matches,
            "train_validation_overlap_count": len(overlap),
            "train_validation_overlap_wsi_ids": list(overlap),
            "masked_holdout_overlap_count": len(masked_overlap),
            "masked_holdout_overlap_wsi_ids": list(masked_overlap),
            "masked_holdout_csv": "docs/data/ubc_ocean_masked_holdout_ids.csv",
            "masked_holdout_id_count": len(holdout_wsi_ids),
            "non_tma_provenance_checked": canonical,
            "non_tma_provenance_source": (
                "docs/behavior_inventory_kaggle.md verified 2026-06-10"
                if canonical
                else "not_checked_for_local_fixture"
            ),
        },
    )


def _window_contract_proof(settings: RealDataRuntimePretestSettings) -> JsonObject:
    train_window_sum = sum(window.patch_count for window in settings.train_windows)
    validation_window_sum = sum(
        window.patch_count for window in settings.validation_windows
    )
    train_cap_matches = train_window_sum == settings.cap_train_patch_count
    validation_cap_matches = (
        validation_window_sum == settings.cap_validation_patch_count
    )
    policy_matches = settings.window_policy == EXPECTED_WINDOW_POLICY
    train_exact = _windows_match(settings.train_windows, EXPECTED_TRAIN_WINDOWS)
    validation_exact = _windows_match(
        settings.validation_windows,
        EXPECTED_VALIDATION_WINDOWS,
    )
    canonical = _canonical_real_counts_and_source(settings)
    status = _real_or_local_contract_status(
        canonical=canonical,
        real_pass=(
            policy_matches
            and train_cap_matches
            and validation_cap_matches
            and settings.cap_train_patch_count == EXPECTED_CAP_TRAIN_PATCH_COUNT
            and settings.cap_validation_patch_count
            == EXPECTED_CAP_VALIDATION_PATCH_COUNT
            and train_exact
            and validation_exact
        ),
        local_pass=policy_matches and train_cap_matches and validation_cap_matches,
    )
    return {
        "status": status,
        "mode": "real" if canonical else "local",
        "window_policy": settings.window_policy,
        "window_policy_matches": policy_matches,
        "train_window_patch_sum": train_window_sum,
        "validation_window_patch_sum": validation_window_sum,
        "train_cap_patch_count": settings.cap_train_patch_count,
        "validation_cap_patch_count": settings.cap_validation_patch_count,
        "train_cap_matches_window_sum": train_cap_matches,
        "validation_cap_matches_window_sum": validation_cap_matches,
        "train_windows_match_locked_real_contract": train_exact,
        "validation_windows_match_locked_real_contract": validation_exact,
        "expected_train_windows": _expected_windows_payload(EXPECTED_TRAIN_WINDOWS),
        "expected_validation_windows": _expected_windows_payload(
            EXPECTED_VALIDATION_WINDOWS,
        ),
    }


def _real_or_local_contract_status(
    *,
    canonical: bool,
    real_pass: bool,
    local_pass: bool,
) -> str:
    if canonical:
        return PASS_STATUS if real_pass else "fail"
    return "local_pass" if local_pass else "fail"


def _canonical_real_contract(settings: RealDataRuntimePretestSettings) -> bool:
    return _canonical_real_counts_and_source(settings) and (
        settings.cap_train_patch_count == EXPECTED_CAP_TRAIN_PATCH_COUNT
        and settings.cap_validation_patch_count == EXPECTED_CAP_VALIDATION_PATCH_COUNT
        and settings.window_policy == EXPECTED_WINDOW_POLICY
        and _windows_match(settings.train_windows, EXPECTED_TRAIN_WINDOWS)
        and _windows_match(settings.validation_windows, EXPECTED_VALIDATION_WINDOWS)
    )


def _canonical_real_counts_and_source(
    settings: RealDataRuntimePretestSettings,
) -> bool:
    return (
        settings.dataset_slug == EXPECTED_DATASET_SLUG
        and settings.real_train_patch_count == EXPECTED_REAL_TRAIN_PATCH_COUNT
        and settings.real_validation_patch_count == EXPECTED_REAL_VALIDATION_PATCH_COUNT
    )


def _windows_match(
    windows: Sequence[WindowSpec],
    expected: Sequence[tuple[str, int, int]],
) -> bool:
    observed = tuple(
        (window.name, window.start_row, window.patch_count) for window in windows
    )
    return observed == tuple(expected)


def _expected_windows_payload(
    windows: Sequence[tuple[str, int, int]],
) -> list[JsonValue]:
    return [
        {
            "name": name,
            "start_row": start_row,
            "patch_count": patch_count,
            "stop_row": start_row + patch_count,
        }
        for name, start_row, patch_count in windows
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _clean_validation_dataloader_proof(  # noqa: PLR0914
    *,
    paths: PatchDataPaths,
    settings: RealDataRuntimePretestSettings,
) -> JsonObject:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    from eqvae.data.dataloaders import normalize_uint8_batch  # noqa: PLC0415
    from eqvae.data.training_batches import (  # noqa: PLC0415
        PatchTrainingDataset,
        PatchTrainingDatasetSpec,
        collate_patch_training_samples,
    )

    validation_indices = _window_indices(settings.validation_windows)
    if not validation_indices:
        message = "validation windows must select at least one row"
        raise ValueError(message)
    batch_size = min(VALIDATION_CLEAN_PROOF_BATCH_SIZE, len(validation_indices))
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.validation.bin_path,
            csv_path=paths.validation.csv_path,
            split=paths.validation.split,
            image_size=settings.image_size,
            channels=settings.channels,
            validate_crc=False,
        ),
    )
    subset = Subset(dataset, validation_indices)
    loader = cast(
        "DataLoader[PatchTrainingBatch]",
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
            collate_fn=collate_patch_training_samples,
        ),
    )
    fetch_ms: list[float] = []
    batches_seen = 0
    samples_seen = 0
    first_batch_hash = ""
    last_batch_hash = ""
    first_sample_hash = ""
    last_sample_hash = ""
    normalized_min = math.inf
    normalized_max = -math.inf
    last_batch_size = 0
    try:
        iterator = iter(loader)
        while True:
            start_ns = time.perf_counter_ns()
            try:
                batch = cast("PatchTrainingBatch", next(iterator))
            except StopIteration:
                break
            fetch_ms.append(_elapsed_ms(start_ns))
            normalized = normalize_uint8_batch(batch.images_uint8)
            if batches_seen == 0:
                first_batch_hash = _hash_sequence(batch.sample_ids)
                first_sample_hash = _hash_text(batch.sample_ids[0])
            last_batch_hash = _hash_sequence(batch.sample_ids)
            last_sample_hash = _hash_text(batch.sample_ids[-1])
            batches_seen += 1
            batch_count = int(batch.images_uint8.shape[0])
            samples_seen += batch_count
            last_batch_size = batch_count
            normalized_min = min(normalized_min, float(torch.amin(normalized).item()))
            normalized_max = max(normalized_max, float(torch.amax(normalized).item()))
    finally:
        dataset.close()
    expected_batches = math.ceil(len(validation_indices) / batch_size)
    status = (
        PASS_STATUS
        if samples_seen == len(validation_indices) and batches_seen == expected_batches
        else "fail"
    )
    return {
        "status": status,
        "split": "validation",
        "dataset_class": "PatchTrainingDataset",
        "collate_fn": "collate_patch_training_samples",
        "normalizer": "normalize_uint8_batch",
        "proof_scope": "validation_loader_clean_input_only",
        "clean_view": "eval_clean",
        "corruption_called": False,
        "corruption_rng_instrumented": False,
        "clean_validation_rng_status": "not_exercised_in_this_loader_lane",
        "clean_validation_rng_consumed": None,
        "num_workers": DEFAULT_DATALOADER_NUM_WORKERS,
        "batch_size": batch_size,
        "expected_sample_count": len(validation_indices),
        "sample_count": samples_seen,
        "expected_batches": expected_batches,
        "batches_seen": batches_seen,
        "last_batch_size": last_batch_size,
        "partial_batch_observed": 0 < last_batch_size < batch_size,
        "images_dtype": "torch.uint8",
        "normalized_dtype": "torch.float32",
        "normalized_min": _format_float(normalized_min),
        "normalized_max": _format_float(normalized_max),
        "normalization_range_pass": normalized_min >= -1.0 and normalized_max <= 1.0,
        "first_batch_sample_id_hash": first_batch_hash,
        "last_batch_sample_id_hash": last_batch_hash,
        "first_sample_id_hash": first_sample_hash,
        "last_sample_id_hash": last_sample_hash,
        "batch_fetch_ms_p50": _format_float(_percentile(fetch_ms, 0.50)),
        "batch_fetch_ms_p95": _format_float(_percentile(fetch_ms, 0.95)),
        "loader_samples_sec": _format_float(
            0.0 if sum(fetch_ms) <= 0.0 else samples_seen / (sum(fetch_ms) / 1000.0),
        ),
    }


def _windows_proof(
    *,
    split: str,
    records: Sequence[PatchRecord],
    windows: Sequence[WindowSpec],
) -> JsonObject:
    window_payloads: list[JsonObject] = []
    selected_records: list[PatchRecord] = []
    status = PASS_STATUS
    for window in windows:
        in_range = 0 <= window.start_row < window.stop_row <= len(records)
        window_records = (
            list(records[window.start_row : window.stop_row]) if in_range else []
        )
        patch_count_matches = len(window_records) == window.patch_count
        if not in_range or not patch_count_matches:
            status = "fail"
        selected_records.extend(window_records)
        window_payloads.append(
            _window_proof_payload(
                split=split,
                window=window,
                records=window_records,
                in_range=in_range,
                patch_count_matches=patch_count_matches,
            ),
        )
    duplicate_semantic_count = _duplicate_semantic_key_count(
        split=split,
        records=selected_records,
    )
    if duplicate_semantic_count:
        status = "fail"
    return cast(
        "JsonObject",
        {
            "status": status,
            "split": split,
            "window_count": len(windows),
            "selected_patch_count": len(selected_records),
            "selected_wsi_count": len({record.wsi_id for record in selected_records}),
            "duplicate_semantic_key_count": duplicate_semantic_count,
            "selected_sample_id_hash": _records_hash(
                split=split,
                records=selected_records,
                identity="sample_id",
            ),
            "selected_semantic_key_hash": _records_hash(
                split=split,
                records=selected_records,
                identity="semantic_key",
            ),
            "windows": window_payloads,
        },
    )


def _window_proof_payload(
    *,
    split: str,
    window: WindowSpec,
    records: Sequence[PatchRecord],
    in_range: bool,
    patch_count_matches: bool,
) -> JsonObject:
    label_counts = Counter(record.label for record in records)
    payload: JsonObject = {
        "name": window.name,
        "split": split,
        "start_row": window.start_row,
        "stop_row": window.stop_row,
        "patch_count": window.patch_count,
        "in_range": in_range,
        "patch_count_matches": patch_count_matches,
        "observed_patch_count": len(records),
        "wsi_count": len({record.wsi_id for record in records}),
        "label_counts": {
            str(label): label_counts[label] for label in sorted(label_counts)
        },
        "sample_id_hash": _records_hash(
            split=split,
            records=records,
            identity="sample_id",
        ),
        "semantic_key_hash": _records_hash(
            split=split,
            records=records,
            identity="semantic_key",
        ),
    }
    if records:
        payload["first_record"] = _record_identity_payload(records[0], split=split)
        payload["last_record"] = _record_identity_payload(records[-1], split=split)
    return payload


def _record_identity_payload(record: PatchRecord, *, split: str) -> JsonObject:
    sample_id = record.sample_id(split)
    semantic_key = _semantic_key(record, split=split)
    return {
        "row_index": record.row_index,
        "file_index": record.file_index,
        "wsi_id": record.wsi_id,
        "label": record.label,
        "x": record.x,
        "y": record.y,
        "sample_id": sample_id,
        "sample_id_hash": _hash_text(sample_id),
        "semantic_sample_key": semantic_key,
        "semantic_sample_key_hash": _hash_text(semantic_key),
    }


def _binary_integrity_payload(
    *,
    bin_path: Path,
    header_size: int,
    expected_crc32: int,
) -> JsonObject:
    hasher = hashlib.sha256()
    checksum = 0
    size_bytes = 0
    with bin_path.open("rb") as binary_file:
        header = binary_file.read(header_size)
        if len(header) != header_size:
            message = f"Expected {header_size}-byte binary header in {bin_path}"
            raise ValueError(message)
        hasher.update(header)
        size_bytes += len(header)
        while True:
            chunk = binary_file.read(FILE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            size_bytes += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
    observed_crc32 = checksum & 0xFFFFFFFF
    return {
        "sha256": hasher.hexdigest(),
        "size_bytes": size_bytes,
        "observed_crc32": observed_crc32,
        "crc_matches_header": observed_crc32 == expected_crc32,
    }


def _expected_patch_count(
    *,
    settings: RealDataRuntimePretestSettings,
    split: str,
) -> int:
    if split == "train":
        return settings.real_train_patch_count
    if split == "validation":
        return settings.real_validation_patch_count
    message = f"Unknown split {split!r}"
    raise ValueError(message)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(FILE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _semantic_key(record: PatchRecord, *, split: str) -> str:
    return f"{split}:{record.wsi_id}:{record.label}:{record.x}:{record.y}"


def _records_hash(
    *,
    split: str,
    records: Sequence[PatchRecord],
    identity: str,
) -> str:
    hasher = hashlib.sha256()
    for record in records:
        if identity == "sample_id":
            value = record.sample_id(split)
        elif identity == "semantic_key":
            value = _semantic_key(record, split=split)
        else:
            message = f"Unsupported identity hash kind: {identity}"
            raise ValueError(message)
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _hash_sequence(values: Sequence[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _duplicate_semantic_key_count(
    *,
    split: str,
    records: Sequence[PatchRecord],
) -> int:
    counts = Counter(_semantic_key(record, split=split) for record in records)
    return sum(count - 1 for count in counts.values() if count > 1)


def _combined_status(*statuses: str) -> str:
    return PASS_STATUS if all(status == PASS_STATUS for status in statuses) else "fail"


def _proof_status_exercised(status: str) -> bool:
    return status in {PASS_STATUS, "local_pass"}


def _required_object_list(payload: JsonObject, key: str) -> list[JsonObject]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected list at {key}"
        raise TypeError(message)
    parsed: list[JsonObject] = []
    for item in value:
        if not isinstance(item, dict):
            message = f"Expected object list at {key}"
            raise TypeError(message)
        parsed.append(cast("JsonObject", item))
    return parsed


def _csv_rows_from_payload(payload: JsonObject, key: str) -> list[dict[str, str]]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            message = f"Expected CSV row objects at {key}"
            raise TypeError(message)
        row: dict[str, str] = {}
        for column, cell in item.items():
            row[str(column)] = "" if cell is None else str(cell)
        rows.append(row)
    return rows


def _linked_status(payload: JsonObject, key: str) -> str:
    return _required_str(_required_object(payload, key), "status")


def _linked_data_ready(data_proof: JsonObject) -> bool:
    return (
        _proof_status_exercised(_required_str(data_proof, "identity_status"))
        and _proof_status_exercised(_required_str(data_proof, "window_status"))
        and _required_str(data_proof, "clean_validation_dataloader_status")
        == PASS_STATUS
        and bool(_required_object(data_proof, "splits"))
    )


def _canonical_or_local_evidence_status(
    *,
    data_proof: JsonObject,
    passed: bool,
) -> str:
    if not passed:
        return FAIL_STATUS
    return (
        PASS_STATUS
        if _required_str(data_proof, "identity_status") == PASS_STATUS
        else LOCAL_PASS_STATUS
    )


def _local_evidence_status(*, passed: bool) -> str:
    return LOCAL_PASS_STATUS if passed else FAIL_STATUS


def _linked_evidence_payload(
    *,
    request: RealDataRuntimePretestRequest,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
    phase_timings: PhaseTimingRecorder,
) -> JsonObject:
    with phase_timings.phase("linked_compile_settle", row_count=len(rows)):
        compile_settle = _compile_settle_proof(settings=settings, rows=rows)
    with phase_timings.phase("linked_ddp_launch", row_count=len(rows)):
        ddp_launch = _ddp_launch_proof(request=request, settings=settings, rows=rows)
    if not _linked_data_ready(data_proof):
        skipped = _linked_evidence_not_run_payload(
            status=SKIPPED_UNSUPPORTED,
            failure_kind="real_data_identity_window_or_clean_loader_not_ready",
        )
        return {
            "status": SKIPPED_UNSUPPORTED,
            "compile_settle": compile_settle,
            "ddp_launch": ddp_launch,
            "dataloader_throughput": skipped,
            "paired_numerical": skipped,
            "corruption_equivalence": skipped,
            "gate_health": skipped,
        }
    try:
        (
            dataloader_throughput,
            paired_numerical,
            corruption_equivalence,
            gate_health,
        ) = _run_linked_evidence_lanes(
            settings=settings,
            data_proof=data_proof,
            rows=rows,
            phase_timings=phase_timings,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        failed = _linked_evidence_not_run_payload(
            status=FAIL_STATUS,
            failure_kind=f"linked_evidence_{type(exc).__name__}",
            failure_message=str(exc),
        )
        return {
            "status": FAIL_STATUS,
            "compile_settle": compile_settle,
            "ddp_launch": ddp_launch,
            "dataloader_throughput": failed,
            "paired_numerical": failed,
            "corruption_equivalence": failed,
            "gate_health": failed,
        }
    lane_statuses = (
        _required_str(compile_settle, "status"),
        _required_str(ddp_launch, "status"),
        _required_str(dataloader_throughput, "status"),
        _required_str(paired_numerical, "status"),
        _required_str(corruption_equivalence, "status"),
        _required_str(gate_health, "status"),
    )
    if all(status == PASS_STATUS for status in lane_statuses):
        status = PASS_STATUS
    elif any(status == FAIL_STATUS for status in lane_statuses):
        status = FAIL_STATUS
    elif all(status in {PASS_STATUS, LOCAL_PASS_STATUS} for status in lane_statuses):
        status = LOCAL_PASS_STATUS
    else:
        status = SKIPPED_UNSUPPORTED
    return {
        "status": status,
        "compile_settle": compile_settle,
        "ddp_launch": ddp_launch,
        "dataloader_throughput": dataloader_throughput,
        "paired_numerical": paired_numerical,
        "corruption_equivalence": corruption_equivalence,
        "gate_health": gate_health,
    }


def _run_linked_evidence_lanes(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
    phase_timings: PhaseTimingRecorder,
) -> tuple[JsonObject, JsonObject, JsonObject, JsonObject]:
    with phase_timings.phase("linked_train_step_evidence", row_count=len(rows)):
        train_step_evidence = _paired_train_step_evidence(
            settings=settings,
            data_proof=data_proof,
            rows=rows,
        )
    with phase_timings.phase("linked_dataloader_throughput", row_count=len(rows)):
        dataloader_throughput = _dataloader_throughput_proof(
            settings=settings,
            data_proof=data_proof,
            rows=rows,
        )
    with phase_timings.phase("linked_paired_numerical", row_count=len(rows)):
        paired_numerical = _paired_numerical_proof(
            settings=settings,
            data_proof=data_proof,
            rows=rows,
            train_step_evidence=train_step_evidence,
        )
    with phase_timings.phase("linked_corruption_equivalence", row_count=len(rows)):
        corruption_equivalence = _corruption_equivalence_proof(
            settings=settings,
            data_proof=data_proof,
            rows=rows,
            train_step_evidence=train_step_evidence,
        )
    with phase_timings.phase("linked_gate_health", row_count=len(rows)):
        gate_health = _gate_health_proof(
            data_proof=data_proof,
            train_step_evidence=train_step_evidence,
        )
    return dataloader_throughput, paired_numerical, corruption_equivalence, gate_health


def _linked_evidence_not_run_payload(
    *,
    status: str,
    failure_kind: str,
    failure_message: str | None = None,
) -> JsonObject:
    return {
        "status": status,
        "rows": [],
        "candidate_evidence_count": 0,
        "failed_candidate_evidence": [],
        "failed_candidate_evidence_count": 0,
        "failure_kind": failure_kind,
        "failure_message_hash": _hash_text(failure_message or failure_kind),
        "notes": failure_kind,
    }


def _compile_settle_proof(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    compiled_rows = [row for row in rows if row["compile_scope"] != COMPILE_NONE]
    measured_compiled_rows = [
        row
        for row in compiled_rows
        if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}
    ]
    counter_source_available, counter_snapshot = _dynamo_counter_snapshot()
    measured_counter_rows = [
        row
        for row in measured_compiled_rows
        if _csv_has_int(row, "graph_break_count")
        and _csv_has_int(row, "recompile_count")
    ]
    settle_coverage_pass = False
    configured_pass = (
        settings.compile_settle_steps == REQUIRED_COMPILE_SETTLE_STEPS
        and COMPILE_NONE in settings.compile_scopes
        and COMPILE_MODEL_FORWARD in settings.compile_scopes
        # COMPILE_STEP is implemented + measured (see implemented_compile_scopes), so
        # the grid-completeness contract requires it, like model_forward.
        and COMPILE_STEP in settings.compile_scopes
        and "model_loss" in settings.compile_scopes
        and "train_step_no_optimizer" in settings.compile_scopes
        and counter_source_available
    )
    post_settle_graph_break_count = sum(
        _csv_int_or_zero(row, "graph_break_count") for row in measured_counter_rows
    )
    post_settle_recompile_count = sum(
        _csv_int_or_zero(row, "recompile_count") for row in measured_counter_rows
    )
    measured_pass = (
        bool(measured_counter_rows)
        and len(measured_counter_rows) == len(measured_compiled_rows)
        and settle_coverage_pass
        and post_settle_graph_break_count == 0
        and post_settle_recompile_count == 0
    )
    status = (
        PASS_STATUS
        if measured_pass
        else SKIPPED_UNSUPPORTED
        if configured_pass
        else FAIL_STATUS
    )
    contract_status = LOCAL_PASS_STATUS if configured_pass else FAIL_STATUS
    return {
        "status": status,
        "contract_status": contract_status,
        "proof_scope": "compile_settle_contract_and_counter_source",
        "compile_settle_steps": settings.compile_settle_steps,
        "configured_compile_scopes": list(settings.compile_scopes),
        "counter_source": "torch._dynamo.utils.counters_with_reset_per_row",
        "counter_source_available": counter_source_available,
        "counter_snapshot": counter_snapshot,
        "compiled_row_count": len(compiled_rows),
        "measured_compiled_row_count": len(measured_compiled_rows),
        "measured_counter_row_count": len(measured_counter_rows),
        "child_counter_source_available_row_count": len(measured_counter_rows),
        "compile_settle_coverage_status": SKIPPED_UNSUPPORTED,
        "compile_settle_coverage_pass": settle_coverage_pass,
        "missing_coverage": [
            "clean_validation_step",
            "ddp_rank_path",
            "final_partial_batch_path",
            "mask_cardinality_0",
            "mask_cardinality_1",
            "mask_cardinality_many",
            "mask_cardinality_all",
        ],
        "measurement_status": PASS_STATUS if measured_pass else SKIPPED_UNSUPPORTED,
        "post_settle_graph_break_count": post_settle_graph_break_count,
        "post_settle_recompile_count": post_settle_recompile_count,
        "measured_compiled_row_ids": [row["row_id"] for row in measured_counter_rows],
        "measured_compiled_rows_required_for_canonical_pass": True,
        "child_counter_source_required_for_canonical_pass": True,
        "full_compile_settle_coverage_required_for_canonical_pass": True,
        "canonical_pass_requires_measured_counter_deltas": True,
        "notes": (
            "The model_forward and whole-step (step) compile scopes are measured in "
            "this pretest. Canonical pass requires measured compiled rows with "
            "child-process Dynamo counter availability, full settle-path coverage, and "
            "zero post-settle graph breaks/recompiles. Full coverage remains "
            "implementation-pending, so compiled rows stay ineligible."
        ),
    }


def _dynamo_counter_snapshot() -> tuple[bool, JsonObject]:
    available = _reset_dynamo_counters()
    return available, _dynamo_counter_summary() if available else {}


def _reset_dynamo_counters() -> bool:
    try:
        dynamo_utils = importlib.import_module("torch._dynamo.utils")
    except (ImportError, AttributeError):
        return False
    counters_object = getattr(dynamo_utils, "counters", None)
    if not isinstance(counters_object, MutableMapping):
        return False
    counters = cast("MutableMapping[object, object]", counters_object)
    counters.clear()
    return True


def _dynamo_counter_summary() -> JsonObject:
    try:
        dynamo_utils = importlib.import_module("torch._dynamo.utils")
    except (ImportError, AttributeError):
        return {}
    counters_object = getattr(dynamo_utils, "counters", None)
    if not isinstance(counters_object, MutableMapping):
        return {}
    counters = cast("MutableMapping[object, object]", counters_object)
    return {
        str(key): _json_safe_counter_value(value) for key, value in counters.items()
    }


def _json_safe_counter_value(value: object) -> JsonValue:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, MutableMapping):
        return {
            str(key): _json_safe_counter_value(item)
            for key, item in cast("MutableMapping[object, object]", value).items()
        }
    return str(value)


def _counter_total(payload: JsonObject, needle: str) -> int:
    total = 0
    lowered = needle.lower()
    for key, value in payload.items():
        key_matches = lowered in key.lower()
        if isinstance(value, int) and not isinstance(value, bool):
            if key_matches:
                total += value
        elif isinstance(value, dict):
            nested = cast("JsonObject", value)
            nested_total = _counter_total(nested, needle)
            if key_matches:
                total += _counter_numeric_total(nested)
            total += nested_total
    return total


def _counter_numeric_total(payload: JsonObject) -> int:
    total = 0
    for value in payload.values():
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
        elif isinstance(value, dict):
            total += _counter_numeric_total(cast("JsonObject", value))
    return total


def _ddp_launch_proof(
    *,
    request: RealDataRuntimePretestRequest,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    dual_rows = [row for row in rows if row["accelerator_mode"] == DUAL_T4_DDP]
    world_size_ok = all(row["world_size"] == "2" for row in dual_rows)
    nproc_ok = all(row["nproc_per_node"] == "2" for row in dual_rows)
    contract_pass = bool(dual_rows) and world_size_ok and nproc_ok
    if not contract_pass:
        return {
            "status": FAIL_STATUS,
            "contract_status": FAIL_STATUS,
            "proof_scope": "dual_t4_ddp_launch_contract",
            "configured_dual_row_count": len(dual_rows),
            "measured_dual_row_count": 0,
            "world_size_configured": 2,
            "nproc_per_node_configured": 2,
            "world_size_contract_matches": world_size_ok,
            "nproc_per_node_contract_matches": nproc_ok,
            "launch_executed": False,
            "measurement_status": FAIL_STATUS,
            "canonical_pass_requires_two_visible_t4_ranks": True,
            "rank_assignments": [],
            "notes": "Dual-T4 DDP rows are not configured correctly.",
        }

    launch = _run_ddp_launch_probe(
        request=request,
        settings=settings,
        row_spec=_first_dual_row_spec(settings),
    )
    rank_assignments = _required_object_list(launch, "rank_assignments")
    rank_order = [_required_int(rank, "rank") for rank in rank_assignments]
    local_rank_order = [_required_int(rank, "local_rank") for rank in rank_assignments]
    canonical_pass = (
        _required_str(launch, "status") == PASS_STATUS
        and _required_int(launch, "torchrun_returncode") == 0
        and _required_int(launch, "rank_count") == DUAL_T4_DEVICE_COUNT
        and rank_order == [0, 1]
        and local_rank_order == [0, 1]
        and all("T4" in _required_str(rank, "device_name") for rank in rank_assignments)
        and all(
            _required_int(rank, "world_size") == DUAL_T4_DEVICE_COUNT
            for rank in rank_assignments
        )
        and all(
            _required_int(rank, "current_device") == _required_int(rank, "local_rank")
            for rank in rank_assignments
        )
    )
    status = (
        PASS_STATUS
        if canonical_pass
        else SKIPPED_UNSUPPORTED
        if _required_str(launch, "status") == SKIPPED_UNSUPPORTED
        else FAIL_STATUS
    )
    return cast(
        "JsonObject",
        {
            "status": status,
            "contract_status": LOCAL_PASS_STATUS,
            "proof_scope": "dual_t4_ddp_launch_contract",
            "configured_dual_row_count": len(dual_rows),
            "measured_dual_row_count": 0,
            "world_size_configured": 2,
            "nproc_per_node_configured": 2,
            "world_size_contract_matches": world_size_ok,
            "nproc_per_node_contract_matches": nproc_ok,
            "launch_executed": _required_bool(launch, "launch_executed"),
            "measurement_status": (
                PASS_STATUS if canonical_pass else SKIPPED_UNSUPPORTED
            ),
            "torchrun_returncode": _required_int(launch, "torchrun_returncode"),
            "rank_count": _required_int(launch, "rank_count"),
            "rank_order": _required_list(launch, "rank_order"),
            "local_rank_order": local_rank_order,
            "rank_assignments": rank_assignments,
            "failure_kind": _required_str(launch, "failure_kind"),
            "failure_message_hash": _required_str(launch, "failure_message_hash"),
            "canonical_pass_requires_two_visible_t4_ranks": True,
            "notes": (
                "Dual-rank torchrun launch proof. Canonical pass requires two "
                "observed T4 ranks and successful child return codes. This proof "
                "does not by itself time DDP train steps."
            ),
        },
    )


def _first_dual_row_spec(settings: RealDataRuntimePretestSettings) -> RowSpec:
    for row_spec in _stage1_row_specs(settings):
        if row_spec.accelerator_mode == DUAL_T4_DDP:
            return row_spec
    message = "No dual_t4_ddp row spec configured"
    raise ValueError(message)


def _run_ddp_launch_probe(
    *,
    request: RealDataRuntimePretestRequest,
    settings: RealDataRuntimePretestSettings,
    row_spec: RowSpec,
) -> JsonObject:
    accelerator = _accelerator_observation()
    accelerator_failure = _accelerator_failure(
        row_spec=row_spec,
        accelerator=accelerator,
    )
    if accelerator_failure is not None:
        _status, failure_kind, failure_message = accelerator_failure
        return {
            "status": SKIPPED_UNSUPPORTED,
            "launch_executed": False,
            "torchrun_returncode": -1,
            "rank_count": 0,
            "rank_order": [],
            "rank_assignments": [],
            "accelerator": accelerator,
            "failure_kind": failure_kind,
            "failure_message_hash": _hash_text(failure_message),
        }
    request.output_dir.mkdir(parents=True, exist_ok=True)
    config = ChildRowConfig(
        config_path=request.config_path,
        output_dir=request.output_dir,
        data_root=settings.data_root,
        row_spec=row_spec,
        settings=settings,
    )
    encoded = _encode_child_config(config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "eqvae.benchmarking.real_data_runtime_pretest",
        "--ddp-rank-row",
        encoded,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = row_spec.cuda_visible_devices
    environment["PYTHONPATH"] = _pythonpath_with_current_sys_path(environment)
    with tempfile.TemporaryDirectory(
        prefix=f"eqvae_real_data_pretest_ddp_{row_spec.row_id}_",
        dir=request.output_dir,
    ) as rank_temp_dir:
        rank_dir = Path(rank_temp_dir)
        environment["EQVAE_REAL_DATA_PRETEST_RANK_DIR"] = str(rank_dir)
        try:
            completed = subprocess.run(  # noqa: S603
                command,
                cwd=request.output_dir,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": FAIL_STATUS,
                "launch_executed": True,
                "torchrun_returncode": -1,
                "rank_count": 0,
                "rank_order": [],
                "rank_assignments": [],
                "accelerator": accelerator,
                "failure_kind": "torchrun_timeout",
                "failure_message_hash": _hash_text(str(exc)),
            }
        if completed.returncode != 0:
            return {
                "status": FAIL_STATUS,
                "launch_executed": True,
                "torchrun_returncode": completed.returncode,
                "rank_count": 0,
                "rank_order": [],
                "rank_assignments": [],
                "accelerator": accelerator,
                "failure_kind": "torchrun_failed",
                "failure_message_hash": _hash_text(completed.stderr[-1000:]),
            }
        try:
            rank_payloads = _load_ddp_rank_payloads(rank_dir=rank_dir)
        except RuntimeError as exc:
            return {
                "status": FAIL_STATUS,
                "launch_executed": True,
                "torchrun_returncode": completed.returncode,
                "rank_count": 0,
                "rank_order": [],
                "rank_assignments": [],
                "accelerator": accelerator,
                "failure_kind": "ddp_rank_payload_missing",
                "failure_message_hash": _hash_text(str(exc)),
            }
    ordered = sorted(rank_payloads, key=lambda payload: _required_int(payload, "rank"))
    return {
        "status": PASS_STATUS,
        "launch_executed": True,
        "torchrun_returncode": completed.returncode,
        "rank_count": len(ordered),
        "rank_order": [_required_int(payload, "rank") for payload in ordered],
        "rank_assignments": [
            {
                "rank": _required_int(payload, "rank"),
                "local_rank": _required_int(payload, "local_rank"),
                "current_device": _required_int(payload, "current_device"),
                "world_size": _required_int(payload, "world_size"),
                "device_name": _required_str(payload, "device_name"),
            }
            for payload in ordered
        ],
        "accelerator": accelerator,
        "failure_kind": "",
        "failure_message_hash": "",
    }


def _run_ddp_rank_row(config: ChildRowConfig) -> None:
    import torch.distributed as dist  # noqa: PLC0415

    rank_dir = Path(os.environ["EQVAE_REAL_DATA_PRETEST_RANK_DIR"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        world_size = int(dist.get_world_size())
        payload: JsonObject = {
            "rank": rank,
            "local_rank": local_rank,
            "current_device": int(torch.cuda.current_device()),
            "world_size": world_size,
            "row_id": config.row_spec.row_id,
            "device_name": torch.cuda.get_device_name(local_rank),
        }
        write_json(rank_dir / f"rank_{rank}.json", payload)
        barrier = cast("Callable[[], object]", dist.barrier)
        barrier()
    finally:
        dist.destroy_process_group()


def _load_ddp_rank_payloads(*, rank_dir: Path) -> tuple[JsonObject, ...]:
    payloads: list[JsonObject] = []
    for rank in range(DUAL_T4_DEVICE_COUNT):
        path = rank_dir / f"rank_{rank}.json"
        if not path.exists():
            message = f"missing DDP rank payload {path}"
            raise RuntimeError(message)
        payloads.append(
            cast("JsonObject", json.loads(path.read_text(encoding="utf-8"))),
        )
    return tuple(payloads)


def _dataloader_throughput_proof(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
) -> JsonObject:
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415

    paths = resolve_patch_data_paths(settings.data_root)
    target_rows = _linked_evidence_target_rows(settings=settings, rows=rows)
    split_rows: list[CsvRow] = []
    for target_row in _unique_dataloader_target_rows(target_rows):
        split_rows.extend((
            _measure_dataloader_split(
                settings=settings,
                data_proof=data_proof,
                split="train",
                bin_path=paths.train.bin_path,
                csv_path=paths.train.csv_path,
                indices=_window_indices(settings.train_windows),
                trainer_rows=rows,
                target_row=target_row,
            ),
            _measure_dataloader_split(
                settings=settings,
                data_proof=data_proof,
                split="validation",
                bin_path=paths.validation.bin_path,
                csv_path=paths.validation.csv_path,
                indices=_window_indices(settings.validation_windows),
                trainer_rows=rows,
                target_row=target_row,
            ),
        ))
    status = _combined_linked_row_status(row["status"] for row in split_rows)
    return cast(
        "JsonObject",
        {
            "status": status,
            "proof_scope": "fixed_window_train_validation_loader_throughput",
            "rows": split_rows,
            "candidate_row_specific": any(
                row["status"] in {PASS_STATUS, INELIGIBLE_STATUS} for row in rows
            ),
            "canonical_pass_requires_candidate_row_grid": True,
            "notes": (
                "Rows use the real PatchTrainingDataset/collate/normalizer path. "
                "Canonical pass requires Kaggle GPU H2D, trainer wait-fraction "
                "evidence, and candidate batch-size/accelerator coverage."
            ),
        },
    )


def _measure_dataloader_split(  # noqa: PLR0913, PLR0914
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    split: Literal["train", "validation"],
    bin_path: Path,
    csv_path: Path,
    indices: Sequence[int],
    trainer_rows: Sequence[CsvRow],
    target_row: CsvRow,
) -> CsvRow:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    from eqvae.data.dataloaders import normalize_uint8_batch  # noqa: PLC0415
    from eqvae.data.training_batches import (  # noqa: PLC0415
        PatchTrainingDataset,
        PatchTrainingDatasetSpec,
        collate_patch_training_samples,
    )

    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    requested_batch_size = int(target_row["per_device_batch_size"])
    batch_size = min(requested_batch_size, len(indices))
    if batch_size <= 0:
        return _dataloader_failure_row(
            settings=settings,
            split=split,
            failure_kind="empty_fixed_window_indices",
            target_row=target_row,
        )
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=bin_path,
            csv_path=csv_path,
            split=split,
            image_size=settings.image_size,
            channels=settings.channels,
            validate_crc=False,
        ),
    )
    subset = Subset(dataset, list(indices))
    loader = cast(
        "DataLoader[PatchTrainingBatch]",
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
            collate_fn=collate_patch_training_samples,
            pin_memory=DEFAULT_DATALOADER_PIN_MEMORY,
        ),
    )
    fetch_ms: list[float] = []
    h2d_ms: list[float] = []
    samples_seen = 0
    batches_seen = 0
    total_batches = math.ceil(len(indices) / batch_size)
    warmup_batches = min(settings.warmup_steps, max(0, total_batches - 1))
    measured_target = min(
        settings.measured_steps,
        max(0, total_batches - warmup_batches),
    )
    try:
        iterator = iter(loader)
        for batch_index in range(warmup_batches + measured_target):
            start_fetch = time.perf_counter_ns()
            batch = cast("PatchTrainingBatch", next(iterator))
            fetch_elapsed = _elapsed_ms(start_fetch)
            normalized = normalize_uint8_batch(batch.images_uint8)
            if batch_index >= warmup_batches:
                fetch_ms.append(fetch_elapsed)
                samples_seen += int(batch.images_uint8.shape[0])
                batches_seen += 1
            if device.type == "cuda":
                start_h2d = time.perf_counter_ns()
                normalized.to(
                    device=device,
                    non_blocking=DEFAULT_DATALOADER_NON_BLOCKING_H2D,
                )
                torch.cuda.synchronize(device)
                if batch_index >= warmup_batches:
                    h2d_ms.append(_elapsed_ms(start_h2d))
    finally:
        dataset.close()
    matching_trainer_rows = _matching_trainer_rows(
        rows=trainer_rows,
        target_row=target_row,
    )
    trainer_p95 = _best_trainer_step_p95(matching_trainer_rows)
    fetch_p95 = _percentile(fetch_ms, 0.95)
    h2d_p95 = _percentile(h2d_ms, 0.95)
    transfer_p95 = fetch_p95 + h2d_p95
    data_wait_fraction = (
        ""
        if trainer_p95 <= 0.0
        else _format_float(min(1.0, transfer_p95 / max(trainer_p95, transfer_p95)))
    )
    loader_samples_sec = (
        0.0 if sum(fetch_ms) <= 0.0 else samples_seen / (sum(fetch_ms) / 1000.0)
    )
    trainer_samples_sec = _best_trainer_samples_sec(matching_trainer_rows)
    canonical = _canonical_real_contract(settings)
    dataloader_margin_pass = not canonical or (
        trainer_p95 > 0.0
        and trainer_samples_sec > 0.0
        and bool(data_wait_fraction)
        and float(data_wait_fraction) <= MAX_DATA_WAIT_FRACTION
        and loader_samples_sec
        >= MIN_LOADER_TRAINER_THROUGHPUT_RATIO * trainer_samples_sec
    )
    passed = (
        batches_seen > 0
        and samples_seen > 0
        and (batch_size == requested_batch_size or not canonical)
        and dataloader_margin_pass
    )
    status = _canonical_or_local_evidence_status(
        data_proof=data_proof,
        passed=passed,
    )
    return {
        "run_name": settings.run_name,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": target_row["accelerator_mode"],
        "machine_shape": "NvidiaTeslaT4",
        "world_size": target_row["world_size"],
        "rank": "0",
        "split": split,
        "num_workers": str(DEFAULT_DATALOADER_NUM_WORKERS),
        "prefetch_factor": DEFAULT_DATALOADER_PREFETCH_FACTOR,
        "pin_memory": _format_bool(value=DEFAULT_DATALOADER_PIN_MEMORY),
        "persistent_workers": _format_bool(value=DEFAULT_DATALOADER_PERSISTENT_WORKERS),
        "non_blocking_h2d": _format_bool(value=DEFAULT_DATALOADER_NON_BLOCKING_H2D),
        "batch_size": str(batch_size),
        "batches_measured": str(batches_seen),
        "batch_fetch_ms_p50": _format_float(_percentile(fetch_ms, 0.50)),
        "batch_fetch_ms_p95": _format_float(fetch_p95),
        "h2d_ms_p50": _format_float(_percentile(h2d_ms, 0.50)) if h2d_ms else "",
        "h2d_ms_p95": _format_float(h2d_p95) if h2d_ms else "",
        "loader_samples_sec": _format_float(loader_samples_sec),
        "trainer_samples_sec": _format_float(trainer_samples_sec)
        if trainer_p95 > 0.0
        else "",
        "data_wait_fraction_p50": data_wait_fraction,
        "data_wait_fraction_p95": data_wait_fraction,
        "rank_sample_count": str(samples_seen),
        "dropped_sample_count": "0",
        "status": status,
        "failure_kind": ""
        if status in {PASS_STATUS, LOCAL_PASS_STATUS}
        else _dataloader_failure_kind(
            exact_batch=batch_size == requested_batch_size,
            trainer_p95=trainer_p95,
            trainer_samples_sec=trainer_samples_sec,
            data_wait_fraction=data_wait_fraction,
            loader_samples_sec=loader_samples_sec,
        ),
    }


def _dataloader_failure_row(
    *,
    settings: RealDataRuntimePretestSettings,
    split: Literal["train", "validation"],
    failure_kind: str,
    target_row: CsvRow,
) -> CsvRow:
    return {
        "run_name": settings.run_name,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": target_row["accelerator_mode"],
        "machine_shape": "NvidiaTeslaT4",
        "world_size": target_row["world_size"],
        "rank": "0",
        "split": split,
        "num_workers": str(DEFAULT_DATALOADER_NUM_WORKERS),
        "prefetch_factor": DEFAULT_DATALOADER_PREFETCH_FACTOR,
        "pin_memory": _format_bool(value=DEFAULT_DATALOADER_PIN_MEMORY),
        "persistent_workers": _format_bool(value=DEFAULT_DATALOADER_PERSISTENT_WORKERS),
        "non_blocking_h2d": _format_bool(value=DEFAULT_DATALOADER_NON_BLOCKING_H2D),
        "batch_size": "",
        "batches_measured": "0",
        "batch_fetch_ms_p50": "",
        "batch_fetch_ms_p95": "",
        "h2d_ms_p50": "",
        "h2d_ms_p95": "",
        "loader_samples_sec": "",
        "trainer_samples_sec": "",
        "data_wait_fraction_p50": "",
        "data_wait_fraction_p95": "",
        "rank_sample_count": "",
        "dropped_sample_count": "0",
        "status": FAIL_STATUS,
        "failure_kind": failure_kind,
    }


def _dataloader_failure_kind(
    *,
    exact_batch: bool,
    trainer_p95: float,
    trainer_samples_sec: float,
    data_wait_fraction: str,
    loader_samples_sec: float,
) -> str:
    if not exact_batch:
        return "dataloader_partial_candidate_batch"
    if trainer_p95 <= 0.0 or trainer_samples_sec <= 0.0:
        return "dataloader_missing_matching_trainer_timing"
    if not data_wait_fraction:
        return "dataloader_wait_fraction_unavailable"
    if float(data_wait_fraction) > MAX_DATA_WAIT_FRACTION:
        return "dataloader_wait_fraction_too_high"
    if loader_samples_sec < MIN_LOADER_TRAINER_THROUGHPUT_RATIO * trainer_samples_sec:
        return "dataloader_loader_throughput_margin_too_low"
    return "dataloader_throughput_failed"


def _paired_train_step_evidence(  # noqa: C901, PLR0914
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
) -> JsonObject:
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset  # noqa: PLC0415

    from eqvae.corruption.stain import profile_from_config  # noqa: PLC0415
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415
    from eqvae.data.training_batches import (  # noqa: PLC0415
        PatchTrainingDataset,
        PatchTrainingDatasetSpec,
        collate_patch_training_samples,
    )

    paths = resolve_patch_data_paths(settings.data_root)
    indices = _window_indices(settings.train_windows)
    if not indices:
        message = "train windows must select at least one row"
        raise ValueError(message)
    target_rows = _unique_train_step_target_rows(
        _linked_evidence_target_rows(settings=settings, rows=rows),
    )
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    profile = profile_from_config(settings.corruption_config)
    evidence_items: list[JsonObject] = []
    failed_evidence_items: list[JsonObject] = []
    dataset = PatchTrainingDataset(
        PatchTrainingDatasetSpec(
            bin_path=paths.train.bin_path,
            csv_path=paths.train.csv_path,
            split=paths.train.split,
            image_size=settings.image_size,
            channels=settings.channels,
            validate_crc=False,
        ),
    )
    try:
        for target_row in target_rows:
            requested_batch_size = int(target_row["per_device_batch_size"])
            batch_size = min(requested_batch_size, len(indices))
            if batch_size <= 0:
                continue
            loader = cast(
                "DataLoader[PatchTrainingBatch]",
                DataLoader(
                    Subset(dataset, indices),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=DEFAULT_DATALOADER_NUM_WORKERS,
                    collate_fn=collate_patch_training_samples,
                ),
            )
            batch_evidence: list[JsonObject] = []
            iterator = iter(loader)
            for batch_index in range(REQUIRED_NUMERICAL_FIXED_BATCHES):
                failure: JsonObject | None = None
                try:
                    batch = cast("PatchTrainingBatch", next(iterator))
                    batch_evidence.append(
                        _paired_fixed_batch_train_step_evidence(
                            settings=settings,
                            data_proof=data_proof,
                            target_row=target_row,
                            batch=batch,
                            device=device,
                            batch_index=batch_index,
                        ),
                    )
                except _CandidateTrainStepEvidenceError as exc:
                    failure = _candidate_train_step_failure_item(
                        target_row=target_row,
                        rows=rows,
                        requested_batch_size=requested_batch_size,
                        observed_batch_size=batch_size,
                        batch_index=batch_index,
                        failure_kind=f"candidate_train_step_{exc.cause_type}",
                        failure_message=exc.cause_message,
                        strategy_attempt=exc.strategy_attempt,
                        target_corruption_strategy=exc.target_corruption_strategy,
                    )
                except (RuntimeError, StopIteration, TypeError, ValueError) as exc:
                    failure = _candidate_train_step_failure_item(
                        target_row=target_row,
                        rows=rows,
                        requested_batch_size=requested_batch_size,
                        observed_batch_size=batch_size,
                        batch_index=batch_index,
                        failure_kind=f"candidate_train_step_{type(exc).__name__}",
                        failure_message=str(exc),
                        strategy_attempt="batch_fetch_or_candidate_setup",
                        target_corruption_strategy=target_row["corruption_strategy"],
                    )
                if failure is not None:
                    failed_evidence_items.append(failure)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
            if len(batch_evidence) < REQUIRED_NUMERICAL_FIXED_BATCHES:
                continue
            primary_batch = batch_evidence[0]
            evidence_items.append(
                cast(
                    "JsonObject",
                    {
                        "row_id": target_row["row_id"],
                        "accelerator_mode": target_row["accelerator_mode"],
                        "world_size": int(target_row["world_size"]),
                        "precision_policy": target_row["precision_policy"],
                        "compile_scope": target_row["compile_scope"],
                        "per_device_batch_size": requested_batch_size,
                        "observed_batch_size": _required_int(
                            primary_batch,
                            "observed_batch_size",
                        ),
                        "observed_batch_count": len(batch_evidence),
                        "split": _required_str(primary_batch, "split"),
                        "sample_id_hash": _required_str(
                            primary_batch,
                            "sample_id_hash",
                        ),
                        "semantic_sample_key_hash": _required_str(
                            primary_batch,
                            "semantic_sample_key_hash",
                        ),
                        "branchless": _required_object(primary_batch, "branchless"),
                        "indexed": _required_object(primary_batch, "indexed"),
                        "batch_evidence": batch_evidence,
                    },
                ),
            )
    finally:
        dataset.close()
    if not evidence_items:
        return _candidate_train_step_failure_payload(
            profile_name=profile.name,
            failed_evidence_items=failed_evidence_items,
        )
    primary = evidence_items[0]
    return cast(
        "JsonObject",
        {
            "status": _canonical_or_local_evidence_status(
                data_proof=data_proof,
                passed=True,
            ),
            "profile_name": profile.name,
            "batch_size": _required_int(primary, "observed_batch_size"),
            "split": _required_str(primary, "split"),
            "sample_id_hash": _required_str(primary, "sample_id_hash"),
            "semantic_sample_key_hash": _required_str(
                primary,
                "semantic_sample_key_hash",
            ),
            "branchless": _required_object(primary, "branchless"),
            "indexed": _required_object(primary, "indexed"),
            "candidate_evidence": evidence_items,
            "candidate_evidence_count": len(evidence_items),
            "failed_candidate_evidence": failed_evidence_items,
            "failed_candidate_evidence_count": len(failed_evidence_items),
        },
    )


def _paired_fixed_batch_train_step_evidence(  # noqa: PLR0913
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    target_row: CsvRow,
    batch: PatchTrainingBatch,
    device: torch.device,
    batch_index: int,
) -> JsonObject:
    from eqvae.data.dataloaders import normalize_uint8_batch  # noqa: PLC0415

    clean = normalize_uint8_batch(batch.images_uint8).to(device=device)
    try:
        branchless = _one_strategy_train_step_evidence(
            settings=settings,
            data_proof=data_proof,
            target_row=target_row,
            clean=clean,
            split=batch.split,
            sample_ids=batch.sample_ids,
            semantic_sample_keys=batch.semantic_sample_keys,
            strategy=BRANCHLESS_ALL,
            device=device,
            capture_gate_rows=True,
            batch_index=batch_index,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _CandidateTrainStepEvidenceError(
            strategy_attempt=BRANCHLESS_ALL,
            target_corruption_strategy=target_row["corruption_strategy"],
            cause=exc,
        ) from None
    try:
        indexed = _one_strategy_train_step_evidence(
            settings=settings,
            data_proof=data_proof,
            target_row=target_row,
            clean=clean,
            split=batch.split,
            sample_ids=batch.sample_ids,
            semantic_sample_keys=batch.semantic_sample_keys,
            strategy=INDEXED_MASKED,
            device=device,
            capture_gate_rows=False,
            batch_index=batch_index,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _CandidateTrainStepEvidenceError(
            strategy_attempt=INDEXED_MASKED,
            target_corruption_strategy=target_row["corruption_strategy"],
            cause=exc,
        ) from None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cast(
        "JsonObject",
        {
            "batch_index": batch_index,
            "observed_batch_size": int(clean.shape[0]),
            "split": batch.split,
            "sample_id_hash": _hash_sequence(batch.sample_ids),
            "semantic_sample_key_hash": _hash_sequence(
                batch.semantic_sample_keys,
            ),
            "branchless": branchless,
            "indexed": indexed,
        },
    )


def _candidate_train_step_failure_item(  # noqa: PLR0913
    *,
    target_row: CsvRow,
    rows: Sequence[CsvRow],
    requested_batch_size: int,
    observed_batch_size: int,
    batch_index: int,
    failure_kind: str,
    failure_message: str,
    strategy_attempt: str,
    target_corruption_strategy: str,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "row_id": target_row["row_id"],
            "accelerator_mode": target_row["accelerator_mode"],
            "world_size": int(target_row["world_size"]),
            "precision_policy": target_row["precision_policy"],
            "compile_scope": target_row["compile_scope"],
            "per_device_batch_size": requested_batch_size,
            "observed_batch_size": observed_batch_size,
            "batch_index": batch_index,
            "strategy_attempt": strategy_attempt,
            "target_corruption_strategy": target_corruption_strategy,
            "affected_row_ids": _matching_train_step_target_row_ids(
                rows=rows,
                target_row=target_row,
            ),
            "status": FAIL_STATUS,
            "failure_kind": failure_kind,
            "failure_message_excerpt": _failure_message_excerpt(failure_message),
            "failure_message_hash": _hash_text(failure_message),
        },
    )


def _candidate_train_step_failure_payload(
    *,
    profile_name: str,
    failed_evidence_items: Sequence[JsonObject],
) -> JsonObject:
    failure_kind = (
        _required_str(failed_evidence_items[0], "failure_kind")
        if failed_evidence_items
        else "candidate_train_step_evidence_unavailable"
    )
    failure_message_hash = (
        _required_str(failed_evidence_items[0], "failure_message_hash")
        if failed_evidence_items
        else _hash_text(failure_kind)
    )
    return {
        "status": FAIL_STATUS,
        "profile_name": profile_name,
        "batch_size": 0,
        "split": "",
        "sample_id_hash": "",
        "semantic_sample_key_hash": "",
        "candidate_evidence": [],
        "candidate_evidence_count": 0,
        "failed_candidate_evidence": list(failed_evidence_items),
        "failed_candidate_evidence_count": len(failed_evidence_items),
        "failure_kind": "candidate_train_step_evidence_unavailable",
        "first_failure_kind": failure_kind,
        "failure_message_hash": failure_message_hash,
    }


def _one_strategy_train_step_evidence(  # noqa: PLR0913, PLR0914
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    target_row: CsvRow,
    clean: torch.Tensor,
    split: str,
    sample_ids: Sequence[str],
    semantic_sample_keys: Sequence[str],
    strategy: str,
    device: torch.device,
    capture_gate_rows: bool,
    batch_index: int,
) -> JsonObject:
    import torch  # noqa: PLC0415

    from eqvae.corruption.stain import (  # noqa: PLC0415
        corrupt_normalized_batch,
        profile_from_config,
    )
    from eqvae.losses.vae import beta_for_step  # noqa: PLC0415
    from eqvae.models.registry import (  # noqa: PLC0415
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        build_model,
    )
    from eqvae.training.optim import (  # noqa: PLC0415
        SpecAdamWConfig,
        create_adamw_optimizer,
    )
    from eqvae.training.step import TrainStepRequest, run_train_step  # noqa: PLC0415

    manual_seed = cast("Callable[[int], torch.Generator]", torch.manual_seed)
    manual_seed(settings.global_seed)
    raw_model = build_model(
        MODEL_KIND_NON_EQ_TRANSLATABLE,
        model_config={"norm_groups": settings.norm_groups},
    ).to(device)
    # Eps/latent shape follows the built model, never a frozen module constant, so a
    # future model with a different latent width re-runs this pretest unchanged
    # (Spec 0011 R1). Read from the raw model before compile wrapping.
    latent_channels = raw_model.latent_channels
    model = _model_for_compile_scope_name(
        model=raw_model,
        compile_scope=target_row["compile_scope"],
    )
    optimizer, _summary = create_adamw_optimizer(
        raw_model,
        config=SpecAdamWConfig(
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            gate_lr_multiplier=1.0,
            gradient_clip_global_norm=settings.gradient_clip_global_norm,
            beta1=0.9,
            beta2=0.999,
        ),
    )
    gate_snapshots = _gate_parameter_snapshots(raw_model)
    gate_captures: dict[str, TensorPayload] = {}
    hooks = (
        _register_gate_capture_hooks(model=raw_model, captures=gate_captures)
        if capture_gate_rows
        else []
    )
    profile = profile_from_config(settings.corruption_config)
    corruption = corrupt_normalized_batch(
        clean,
        profile=profile,
        corruption_seed=settings.corruption_seed,
        split=split,
        semantic_sample_keys=semantic_sample_keys,
        corruption_step=batch_index,
        corruption_view="train_corrupted_real_data_runtime_pretest",
        strategy=strategy,
    )
    shape = cast("tuple[int, int, int, int]", tuple(clean.shape))
    eps = torch.zeros(
        (
            shape[0],
            latent_channels,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
            settings.image_size // LATENT_DOWNSAMPLE_FACTOR,
        ),
        dtype=torch.float32,
        device=device,
    )
    beta = beta_for_step(
        optimizer_step_index=batch_index,
        max_optimizer_steps=settings.warmup_steps + settings.measured_steps,
        target_beta=settings.beta_target,
        warmup_fraction=settings.beta_warmup_fraction,
    )
    result = run_train_step(
        TrainStepRequest(
            model=model,
            optimizer=optimizer,
            clean_batch=clean,
            eps=eps,
            beta=beta,
            ssim_weight=settings.ssim_weight,
            optimizer_step_index=batch_index,
            gradient_clip_global_norm=settings.gradient_clip_global_norm,
            input_batch=corruption.corrupted,
        ),
    )
    for hook in hooks:
        hook.remove()
    loss_scalars = result.losses.detached_scalars()
    gate_rows = (
        _gate_rows_from_model(
            settings=settings,
            data_proof=data_proof,
            model=raw_model,
            snapshots=gate_snapshots,
            captures=gate_captures,
        )
        if capture_gate_rows
        else []
    )
    return cast(
        "JsonObject",
        {
            "status": _canonical_or_local_evidence_status(
                data_proof=data_proof,
                passed=result.nonfinite_count == 0,
            ),
            "strategy": strategy,
            "batch_index": batch_index,
            "losses": loss_scalars,
            "grad_norm": result.grad_norm,
            "param_update_norm": result.param_update_norm,
            "nonfinite_count": result.nonfinite_count,
            "amp_step_skipped": False,
            "x_hat_min": float(result.forward.reconstruction.detach().amin().item()),
            "x_hat_max": float(result.forward.reconstruction.detach().amax().item()),
            "mu_mean": float(result.forward.mu.detach().mean().item()),
            "mu_std": float(result.forward.mu.detach().std(unbiased=False).item()),
            "logvar_mean": float(result.forward.logvar.detach().mean().item()),
            "logvar_std": float(
                result.forward.logvar.detach().std(unbiased=False).item(),
            ),
            "logvar_clamp_count": int(
                result.forward.logvar_clamp_count.detach().item(),
            ),
            "corrupted_hash": _tensor_sha256(corruption.corrupted),
            "stain_only_hash": _tensor_sha256(corruption.stain_only),
            "gaussian_only_hash": _tensor_sha256(corruption.gaussian_only),
            "combined_hash": _tensor_sha256(corruption.combined),
            "metadata_hash": _metadata_hash(
                [metadata.as_json() for metadata in corruption.metadata],
            ),
            "applied_mask_hash": _hash_sequence(
                ["1" if metadata.applied else "0" for metadata in corruption.metadata],
            ),
            "stain_param_hash": _hash_sequence(
                [
                    json.dumps(
                        {"alpha": metadata.alpha, "beta": metadata.beta},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for metadata in corruption.metadata
                ],
            ),
            "noise_std_hash": _hash_sequence(
                [_format_float(metadata.noise_std) for metadata in corruption.metadata],
            ),
            "clean_sample_unchanged_count": _clean_sample_unchanged_count(
                clean=clean,
                corrupted=corruption.corrupted,
                applied=[metadata.applied for metadata in corruption.metadata],
            ),
            "sample_id_hash": _hash_sequence(sample_ids),
            "semantic_sample_key_hash": _hash_sequence(semantic_sample_keys),
            "gate_rows": gate_rows,
        },
    )


def _paired_numerical_proof(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
    train_step_evidence: JsonObject,
) -> JsonObject:
    if _required_str(train_step_evidence, "status") == FAIL_STATUS:
        return _train_step_dependency_failure_proof(
            train_step_evidence=train_step_evidence,
            proof_scope="paired_branchless_indexed_train_step_candidate_batches",
            reference_strategy=BRANCHLESS_ALL,
            candidate_strategy=INDEXED_MASKED,
            notes=(
                "Paired numerical checks could not run because candidate "
                "train-step evidence failed before any full fixed-batch "
                "candidate evidence row was produced."
            ),
        )
    primary_delta = _numerical_delta_payload(
        reference=_required_object(train_step_evidence, "branchless"),
        candidate=_required_object(train_step_evidence, "indexed"),
    )
    csv_rows: list[CsvRow] = []
    for row in rows:
        evidence = _train_step_evidence_for_row(
            row=row,
            train_step_evidence=train_step_evidence,
        )
        reference_evidence = _train_step_reference_evidence_for_row(
            row=row,
            train_step_evidence=train_step_evidence,
        )
        if evidence is None or reference_evidence is None:
            csv_rows.append(
                _numerical_row(
                    settings=settings,
                    row=row,
                    status=SKIPPED_UNSUPPORTED,
                    delta_payload=primary_delta,
                    batch_index=0,
                    reference_row_id=_reference_row_id_for_row(row),
                ),
            )
            continue
        reference_batches = _batch_evidence_items(reference_evidence)
        candidate_batches = _batch_evidence_items(evidence)
        for batch_index, reference_batch in enumerate(reference_batches):
            if batch_index >= len(candidate_batches):
                break
            candidate_batch = candidate_batches[batch_index]
            delta_payload = _numerical_delta_payload(
                reference=_required_object(reference_batch, "branchless"),
                candidate=_strategy_payload_for_row(
                    row=row,
                    evidence_batch=candidate_batch,
                ),
            )
            exact_batch = _evidence_observed_exact_batch(
                row=row,
                evidence=evidence,
            ) and _evidence_observed_exact_batch(
                row=row,
                evidence=reference_evidence,
            )
            status = _canonical_or_local_evidence_status(
                data_proof=data_proof,
                passed=_required_bool(delta_payload, "passed")
                and exact_batch
                and len(reference_batches) >= REQUIRED_NUMERICAL_FIXED_BATCHES
                and len(candidate_batches) >= REQUIRED_NUMERICAL_FIXED_BATCHES,
            )
            csv_rows.append(
                _numerical_row(
                    settings=settings,
                    row=row,
                    status=status,
                    delta_payload=delta_payload,
                    batch_index=batch_index,
                    reference_row_id=_required_str(reference_evidence, "row_id"),
                ),
            )
    candidate_row_specific = any(
        row["status"] in {PASS_STATUS, LOCAL_PASS_STATUS} for row in csv_rows
    )
    lane_status = (
        _combined_linked_row_status(row["status"] for row in csv_rows)
        if candidate_row_specific
        else _local_evidence_status(passed=_required_bool(primary_delta, "passed"))
    )
    return cast(
        "JsonObject",
        {
            "status": lane_status,
            "proof_scope": "paired_branchless_indexed_train_step_candidate_batches",
            "rows": csv_rows,
            "candidate_row_specific": candidate_row_specific,
            "canonical_pass_requires_candidate_batch_grid": True,
            "required_fixed_batch_count": REQUIRED_NUMERICAL_FIXED_BATCHES,
            "reference_strategy": BRANCHLESS_ALL,
            "candidate_strategy": INDEXED_MASKED,
            "batch_size": _required_int(train_step_evidence, "batch_size"),
            "candidate_evidence_count": _required_int(
                train_step_evidence,
                "candidate_evidence_count",
            ),
            "failed_candidate_evidence_count": _required_int(
                train_step_evidence,
                "failed_candidate_evidence_count",
            ),
            "failed_candidate_evidence": _required_object_list(
                train_step_evidence,
                "failed_candidate_evidence",
            ),
            "sample_id_hash": _required_str(train_step_evidence, "sample_id_hash"),
            "semantic_sample_key_hash": _required_str(
                train_step_evidence,
                "semantic_sample_key_hash",
            ),
            "delta_summary": primary_delta,
            "notes": (
                "This proof compares each covered candidate against the "
                "amp_off_fp32 eager branchless_all reference on fixed batches. "
                "DDP candidate numerical checks remain unsupported until real "
                "DDP train-step timing is implemented."
            ),
        },
    )


def _corruption_equivalence_proof(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    rows: Sequence[CsvRow],
    train_step_evidence: JsonObject,
) -> JsonObject:
    if _required_str(train_step_evidence, "status") == FAIL_STATUS:
        payload = _train_step_dependency_failure_proof(
            train_step_evidence=train_step_evidence,
            proof_scope=("branchless_indexed_corruption_equivalence_candidate_batches"),
            notes=(
                "Corruption equivalence could not run because candidate "
                "train-step evidence failed before any full fixed-batch "
                "candidate evidence row was produced."
            ),
        )
        payload["hashes_match"] = False
        payload["clean_validation_rng_status"] = "not_exercised_training_batch_only"
        payload["clean_validation_rng_advanced"] = None
        return payload
    primary_branchless = _required_object(train_step_evidence, "branchless")
    primary_indexed = _required_object(train_step_evidence, "indexed")
    csv_rows: list[CsvRow] = []
    hashes_match_by_row: list[bool] = []
    for row in rows:
        evidence = _train_step_evidence_for_row(
            row=row,
            train_step_evidence=train_step_evidence,
        )
        if evidence is None:
            csv_rows.append(
                _corruption_row(
                    settings=settings,
                    row=row,
                    status=SKIPPED_UNSUPPORTED,
                    batch_index=0,
                    branchless=primary_branchless,
                    indexed=primary_indexed,
                ),
            )
            continue
        for batch in _batch_evidence_items(evidence):
            branchless = _required_object(batch, "branchless")
            indexed = _required_object(batch, "indexed")
            hashes_match = _corruption_hashes_match(
                branchless=branchless,
                indexed=indexed,
            )
            hashes_match_by_row.append(hashes_match)
            status = _canonical_or_local_evidence_status(
                data_proof=data_proof,
                passed=hashes_match
                and _evidence_observed_exact_batch(row=row, evidence=evidence)
                and len(_batch_evidence_items(evidence))
                >= REQUIRED_NUMERICAL_FIXED_BATCHES,
            )
            csv_rows.append(
                _corruption_row(
                    settings=settings,
                    row=row,
                    status=status,
                    batch_index=_required_int(batch, "batch_index"),
                    branchless=branchless,
                    indexed=indexed,
                ),
            )
    candidate_row_specific = any(
        row["status"] in {PASS_STATUS, LOCAL_PASS_STATUS} for row in csv_rows
    )
    primary_hashes_match = _corruption_hashes_match(
        branchless=primary_branchless,
        indexed=primary_indexed,
    )
    lane_status = (
        _combined_linked_row_status(row["status"] for row in csv_rows)
        if candidate_row_specific
        else _local_evidence_status(passed=primary_hashes_match)
    )
    return cast(
        "JsonObject",
        {
            "status": lane_status,
            "proof_scope": (
                "branchless_indexed_corruption_equivalence_candidate_batches"
            ),
            "rows": csv_rows,
            "candidate_row_specific": candidate_row_specific,
            "canonical_pass_requires_candidate_batch_grid": True,
            "required_fixed_batch_count": REQUIRED_NUMERICAL_FIXED_BATCHES,
            "candidate_evidence_count": _required_int(
                train_step_evidence,
                "candidate_evidence_count",
            ),
            "failed_candidate_evidence_count": _required_int(
                train_step_evidence,
                "failed_candidate_evidence_count",
            ),
            "failed_candidate_evidence": _required_object_list(
                train_step_evidence,
                "failed_candidate_evidence",
            ),
            "hashes_match": (
                all(hashes_match_by_row)
                if candidate_row_specific
                else primary_hashes_match
            ),
            "branchless_corrupted_hash": _required_str(
                primary_branchless,
                "corrupted_hash",
            ),
            "indexed_corrupted_hash": _required_str(
                primary_indexed,
                "corrupted_hash",
            ),
            "metadata_hash": _required_str(primary_branchless, "metadata_hash"),
            "clean_validation_rng_status": "not_exercised_training_batch_only",
            "clean_validation_rng_advanced": None,
            "notes": (
                "Corruption equivalence uses the same stateless semantic keys and "
                "compares branchless_all against indexed_masked for candidate "
                "training batches. It does not exercise the clean-validation "
                "RNG non-consumption lane."
            ),
        },
    )


def _gate_health_proof(
    *,
    data_proof: JsonObject,
    train_step_evidence: JsonObject,
) -> JsonObject:
    if _required_str(train_step_evidence, "status") == FAIL_STATUS:
        failed_items = _required_object_list(
            train_step_evidence,
            "failed_candidate_evidence",
        )
        return cast(
            "JsonObject",
            {
                "status": FAIL_STATUS,
                "proof_scope": "one_train_step_gate_parameter_and_activation_health",
                "rows": [],
                "row_statuses": [],
                "nonfinite_count": len(failed_items),
                "candidate_evidence_count": _required_int(
                    train_step_evidence,
                    "candidate_evidence_count",
                ),
                "failed_candidate_evidence_count": _required_int(
                    train_step_evidence,
                    "failed_candidate_evidence_count",
                ),
                "failed_candidate_evidence": failed_items,
                "failure_kind": _required_str(
                    train_step_evidence,
                    "failure_kind",
                ),
                "notes": (
                    "Gate health could not run because candidate train-step "
                    "evidence failed before any full fixed-batch candidate "
                    "evidence row was produced."
                ),
            },
        )
    evidence_items = _required_object_list(train_step_evidence, "candidate_evidence")
    rows: list[CsvRow] = []
    row_statuses: list[JsonObject] = []
    for evidence in evidence_items:
        evidence_row_id = _required_str(evidence, "row_id")
        branchless_evidence = _required_object(evidence, "branchless")
        evidence_rows = _csv_rows_from_payload(branchless_evidence, "gate_rows")
        evidence_passed = bool(evidence_rows) and all(
            row["gate_health_status"] in {PASS_STATUS, LOCAL_PASS_STATUS}
            for row in evidence_rows
        )
        evidence_status = _canonical_or_local_evidence_status(
            data_proof=data_proof,
            passed=evidence_passed,
        )
        for row in evidence_rows:
            row["candidate_row_id"] = evidence_row_id
            row["row_id"] = f"{evidence_row_id}__gate__{row['module']}"
            if row["gate_health_status"] in {PASS_STATUS, LOCAL_PASS_STATUS}:
                row["gate_health_status"] = evidence_status
        rows.extend(evidence_rows)
        row_statuses.append({
            "row_id": _required_str(evidence, "row_id"),
            "accelerator_mode": _required_str(evidence, "accelerator_mode"),
            "world_size": _required_int(evidence, "world_size"),
            "per_device_batch_size": _required_int(evidence, "per_device_batch_size"),
            "precision_policy": _required_str(evidence, "precision_policy"),
            "compile_scope": _required_str(evidence, "compile_scope"),
            "status": evidence_status,
        })
    nonfinite_count = sum(
        1
        for row in rows
        if row["gate_health_status"] not in {PASS_STATUS, LOCAL_PASS_STATUS}
    )
    lane_status = _combined_linked_row_status(
        _required_str(row_status, "status") for row_status in row_statuses
    )
    return cast(
        "JsonObject",
        {
            "status": lane_status,
            "proof_scope": "one_train_step_gate_parameter_and_activation_health",
            "rows": rows,
            "row_statuses": row_statuses,
            "nonfinite_count": nonfinite_count,
            "notes": (
                "Gate health is measured on the same fixed local/capped batch as "
                "the paired numerical proof and linked back to candidate row "
                "identity. Canonical pass still depends on the canonical "
                "real-data evidence context."
            ),
        },
    )


def _train_step_dependency_failure_proof(
    *,
    train_step_evidence: JsonObject,
    proof_scope: str,
    notes: str,
    reference_strategy: str | None = None,
    candidate_strategy: str | None = None,
) -> JsonObject:
    payload = cast(
        "JsonObject",
        {
            "status": FAIL_STATUS,
            "proof_scope": proof_scope,
            "rows": [],
            "candidate_row_specific": False,
            "canonical_pass_requires_candidate_batch_grid": True,
            "required_fixed_batch_count": REQUIRED_NUMERICAL_FIXED_BATCHES,
            "batch_size": _required_int(train_step_evidence, "batch_size"),
            "candidate_evidence_count": _required_int(
                train_step_evidence,
                "candidate_evidence_count",
            ),
            "failed_candidate_evidence_count": _required_int(
                train_step_evidence,
                "failed_candidate_evidence_count",
            ),
            "failed_candidate_evidence": _required_object_list(
                train_step_evidence,
                "failed_candidate_evidence",
            ),
            "failure_kind": _required_str(train_step_evidence, "failure_kind"),
            "first_failure_kind": _required_str(
                train_step_evidence,
                "first_failure_kind",
            ),
            "failure_message_hash": _required_str(
                train_step_evidence,
                "failure_message_hash",
            ),
            "sample_id_hash": _required_str(train_step_evidence, "sample_id_hash"),
            "semantic_sample_key_hash": _required_str(
                train_step_evidence,
                "semantic_sample_key_hash",
            ),
            "notes": notes,
        },
    )
    if reference_strategy is not None:
        payload["reference_strategy"] = reference_strategy
    if candidate_strategy is not None:
        payload["candidate_strategy"] = candidate_strategy
    return payload


def _combined_linked_row_status(statuses: Sequence[str] | Iterator[str]) -> str:
    parsed = tuple(statuses)
    if parsed and all(status == PASS_STATUS for status in parsed):
        return PASS_STATUS
    if parsed and all(status in {PASS_STATUS, LOCAL_PASS_STATUS} for status in parsed):
        return LOCAL_PASS_STATUS
    if any(status == FAIL_STATUS for status in parsed):
        return FAIL_STATUS
    return SKIPPED_UNSUPPORTED


def _train_step_evidence_for_row(
    *,
    row: CsvRow,
    train_step_evidence: JsonObject,
) -> JsonObject | None:
    for evidence in _required_object_list(train_step_evidence, "candidate_evidence"):
        if (
            _required_str(evidence, "accelerator_mode") == row["accelerator_mode"]
            and str(_required_int(evidence, "world_size")) == row["world_size"]
            and str(_required_int(evidence, "per_device_batch_size"))
            == row["per_device_batch_size"]
            and _required_str(evidence, "precision_policy") == row["precision_policy"]
            and _required_str(evidence, "compile_scope") == row["compile_scope"]
        ):
            return evidence
    return None


def _train_step_reference_evidence_for_row(
    *,
    row: CsvRow,
    train_step_evidence: JsonObject,
) -> JsonObject | None:
    reference_row = dict(row)
    reference_row["compile_scope"] = COMPILE_NONE
    reference_row["corruption_strategy"] = BRANCHLESS_ALL
    for evidence in _required_object_list(train_step_evidence, "candidate_evidence"):
        if (
            _required_str(evidence, "accelerator_mode")
            == reference_row["accelerator_mode"]
            and str(_required_int(evidence, "world_size"))
            == reference_row["world_size"]
            and str(_required_int(evidence, "per_device_batch_size"))
            == reference_row["per_device_batch_size"]
            and _required_str(evidence, "precision_policy")
            == reference_row["precision_policy"]
            and _required_str(evidence, "compile_scope")
            == reference_row["compile_scope"]
        ):
            return evidence
    return None


def _batch_evidence_items(evidence: JsonObject) -> list[JsonObject]:
    return _required_object_list(evidence, "batch_evidence")


def _strategy_payload_for_row(*, row: CsvRow, evidence_batch: JsonObject) -> JsonObject:
    strategy = (
        "indexed" if row["corruption_strategy"] == INDEXED_MASKED else "branchless"
    )
    return _required_object(evidence_batch, strategy)


def _reference_row_id_for_row(row: CsvRow) -> str:
    return _row_id(
        accelerator_mode=row["accelerator_mode"],
        per_device_batch_size=int(row["per_device_batch_size"]),
        precision_policy=row["precision_policy"],
        compile_scope=COMPILE_NONE,
        corruption_strategy=BRANCHLESS_ALL,
    )


def _evidence_observed_exact_batch(*, row: CsvRow, evidence: JsonObject) -> bool:
    return (
        str(_required_int(evidence, "observed_batch_size"))
        == row["per_device_batch_size"]
    )


def _corruption_hashes_match(*, branchless: JsonObject, indexed: JsonObject) -> bool:
    return (
        _required_str(branchless, "corrupted_hash")
        == _required_str(indexed, "corrupted_hash")
        and _required_str(branchless, "metadata_hash")
        == _required_str(indexed, "metadata_hash")
        and _required_str(branchless, "applied_mask_hash")
        == _required_str(indexed, "applied_mask_hash")
        and _required_str(branchless, "stain_param_hash")
        == _required_str(indexed, "stain_param_hash")
        and _required_str(branchless, "noise_std_hash")
        == _required_str(indexed, "noise_std_hash")
    )


def _linked_evidence_target_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
) -> list[CsvRow]:
    measured_rows = [
        row for row in rows if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}
    ]
    if measured_rows:
        return measured_rows
    row_spec = RowSpec(
        row_id=_row_id(
            accelerator_mode=SINGLE_VISIBLE_T4,
            per_device_batch_size=LOCAL_LINKED_EVIDENCE_BATCH_SIZE,
            precision_policy=AMP_OFF_FP32,
            compile_scope=COMPILE_NONE,
            corruption_strategy=BRANCHLESS_ALL,
        ),
        accelerator_mode=SINGLE_VISIBLE_T4,
        per_device_batch_size=LOCAL_LINKED_EVIDENCE_BATCH_SIZE,
        precision_policy=AMP_OFF_FP32,
        compile_scope=COMPILE_NONE,
        corruption_strategy=BRANCHLESS_ALL,
        parent_synthetic_row_id="",
        candidate_role="local_linked_evidence_fixture",
        world_size=1,
        nproc_per_node=1,
        cuda_visible_devices="0",
    )
    row = dict(_base_row(settings=settings, row_spec=row_spec))
    row["status"] = LOCAL_PASS_STATUS
    return [row]


def _unique_dataloader_target_rows(rows: Sequence[CsvRow]) -> list[CsvRow]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[CsvRow] = []
    for row in rows:
        key = (row["accelerator_mode"], row["world_size"], row["per_device_batch_size"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _unique_train_step_target_rows(rows: Sequence[CsvRow]) -> list[CsvRow]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[CsvRow] = []
    for row in rows:
        if row["accelerator_mode"] != SINGLE_VISIBLE_T4:
            continue
        if row["precision_policy"] != AMP_OFF_FP32:
            continue
        if row["compile_scope"] not in {
            COMPILE_NONE,
            COMPILE_MODEL_FORWARD,
            COMPILE_STEP,
        }:
            continue
        key = (
            row["accelerator_mode"],
            row["world_size"],
            row["per_device_batch_size"],
            row["precision_policy"],
            row["compile_scope"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    if unique:
        return sorted(unique, key=_train_step_evidence_priority)
    return [
        row
        for row in rows
        if row["accelerator_mode"] == SINGLE_VISIBLE_T4
        and row["precision_policy"] == AMP_OFF_FP32
        and row["compile_scope"] in {COMPILE_NONE, COMPILE_MODEL_FORWARD, COMPILE_STEP}
    ][:1]


def _train_step_evidence_priority(row: CsvRow) -> tuple[int, int, str]:
    return (
        0 if row["compile_scope"] == COMPILE_NONE else 1,
        int(row["per_device_batch_size"]),
        row["row_id"],
    )


def _matching_train_step_target_row_ids(
    *,
    rows: Sequence[CsvRow],
    target_row: CsvRow,
) -> list[str]:
    return [
        row["row_id"]
        for row in rows
        if row["accelerator_mode"] == target_row["accelerator_mode"]
        and row["world_size"] == target_row["world_size"]
        and row["per_device_batch_size"] == target_row["per_device_batch_size"]
        and row["precision_policy"] == target_row["precision_policy"]
        and row["compile_scope"] == target_row["compile_scope"]
    ]


def _matching_trainer_rows(
    *,
    rows: Sequence[CsvRow],
    target_row: CsvRow,
) -> list[CsvRow]:
    return [
        row
        for row in rows
        if row["accelerator_mode"] == target_row["accelerator_mode"]
        and row["world_size"] == target_row["world_size"]
        and row["per_device_batch_size"] == target_row["per_device_batch_size"]
    ]


def _best_trainer_step_p95(rows: Sequence[CsvRow]) -> float:
    values = [
        _csv_float_or_inf(row, "steady_step_ms_p95")
        for row in rows
        if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}
    ]
    finite = [value for value in values if math.isfinite(value)]
    return min(finite) if finite else 0.0


def _best_trainer_samples_sec(rows: Sequence[CsvRow]) -> float:
    values = [
        _csv_float_or_inf(row, "trainer_samples_sec")
        for row in rows
        if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}
    ]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else 0.0


def _gate_parameter_snapshots(model: NonEquivariantVAE) -> dict[str, TensorPayload]:
    snapshots: dict[str, TensorPayload] = {}
    for name, module in _named_gate_modules(model):
        snapshots[name] = {
            "a": module.a.detach().clone(),
            "b": module.b.detach().clone(),
        }
    return snapshots


def _register_gate_capture_hooks(
    *,
    model: NonEquivariantVAE,
    captures: dict[str, TensorPayload],
) -> list[RemovableHandle]:
    def make_hook(name: str) -> Callable[..., None]:
        def hook(_module: object, inputs: tuple[object, ...], output: object) -> None:
            if name in captures or not inputs:
                return
            input_tensor = inputs[0]
            if not _is_tensor_like(input_tensor) or not _is_tensor_like(output):
                return
            input_t = cast("torch.Tensor", input_tensor)
            output_t = cast("torch.Tensor", output)
            captures[name] = {
                "input": input_t.detach().cpu(),
                "output": output_t.detach().cpu(),
            }

        return hook

    handles: list[RemovableHandle] = []
    for name, module in _named_gate_modules(model):
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def _is_tensor_like(value: object) -> bool:
    import torch  # noqa: PLC0415

    return isinstance(value, torch.Tensor)


def _gate_rows_from_model(  # noqa: PLR0914
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    model: NonEquivariantVAE,
    snapshots: Mapping[str, TensorPayload],
    captures: Mapping[str, TensorPayload],
) -> list[CsvRow]:
    import torch  # noqa: PLC0415

    rows: list[CsvRow] = []
    del data_proof
    status = _local_evidence_status(passed=True)
    for name, module in _named_gate_modules(model):
        snapshot = snapshots[name]
        a_before = snapshot["a"].detach().cpu().to(torch.float32)
        b_before = snapshot["b"].detach().cpu().to(torch.float32)
        capture = captures.get(name)
        input_tensor = (
            capture["input"].to(torch.float32)
            if capture is not None
            else torch.empty((0,), dtype=torch.float32)
        )
        output_tensor = (
            capture["output"].to(torch.float32)
            if capture is not None
            else torch.empty((0,), dtype=torch.float32)
        )
        gate = _gate_values(input_tensor, a_before, b_before)
        channel_output_rms = _channel_rms(output_tensor)
        dead_channel_count = int(
            torch.count_nonzero(channel_output_rms <= GATE_DEAD_RMS_THRESHOLD).item(),
        )
        a_grad = (
            module.a.grad.detach().cpu().to(torch.float32)
            if module.a.grad is not None
            else None
        )
        b_grad = (
            module.b.grad.detach().cpu().to(torch.float32)
            if module.b.grad is not None
            else None
        )
        a_after = module.a.detach().cpu().to(torch.float32)
        b_after = module.b.detach().cpu().to(torch.float32)
        max_abs_a = float(torch.amax(torch.abs(a_after)).item())
        max_abs_b = float(torch.amax(torch.abs(b_after)).item())
        frac_low = _tensor_fraction(gate < GATE_SATURATION_LOW)
        frac_high = _tensor_fraction(gate > GATE_SATURATION_HIGH)
        worst_frac_low = _worst_channel_fraction(gate < GATE_SATURATION_LOW)
        worst_frac_high = _worst_channel_fraction(gate > GATE_SATURATION_HIGH)
        input_rms = _tensor_rms(input_tensor)
        output_rms = _tensor_rms(output_tensor)
        output_input_ratio = _safe_ratio(output_rms, input_rms)
        a_grad_norm = _optional_tensor_norm(a_grad)
        b_grad_norm = _optional_tensor_norm(b_grad)
        finite_pass = (
            torch.isfinite(a_after).all()
            and torch.isfinite(b_after).all()
            and (gate.numel() == 0 or torch.isfinite(gate).all())
        )
        row_status = _gate_health_status_from_metrics(
            base_status=status,
            finite_pass=bool(finite_pass),
            max_abs_a=max_abs_a,
            max_abs_b=max_abs_b,
            max_saturation_fraction=max(frac_low, frac_high),
            worst_channel_saturation_fraction=max(worst_frac_low, worst_frac_high),
            dead_channel_count=dead_channel_count,
            num_elements=int(gate.numel()),
            num_channels=module.channels,
            output_input_rms_ratio=output_input_ratio,
        )
        rows.append({
            "run_name": settings.run_name,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": "false",
            "accelerator_mode": SINGLE_VISIBLE_T4,
            "machine_shape": "NvidiaTeslaT4",
            "optimizer_step": "0",
            "module": name,
            "gate_kind": "scalar_sigmoid_ab",
            "num_channels": str(module.channels),
            "num_elements": str(int(gate.numel())),
            "a_min": _format_float(float(torch.amin(a_after).item())),
            "a_max": _format_float(float(torch.amax(a_after).item())),
            "a_mean": _format_float(float(torch.mean(a_after).item())),
            "a_std": _format_float(float(torch.std(a_after, unbiased=False).item())),
            "b_min": _format_float(float(torch.amin(b_after).item())),
            "b_max": _format_float(float(torch.amax(b_after).item())),
            "b_mean": _format_float(float(torch.mean(b_after).item())),
            "b_std": _format_float(float(torch.std(b_after, unbiased=False).item())),
            "max_abs_a": _format_float(max_abs_a),
            "max_abs_b": _format_float(max_abs_b),
            "gate_mean": _format_float(_tensor_mean(gate)),
            "gate_std": _format_float(_tensor_std(gate)),
            "gate_p01": _format_float(_tensor_quantile(gate, 0.01)),
            "gate_p50": _format_float(_tensor_quantile(gate, 0.50)),
            "gate_p99": _format_float(_tensor_quantile(gate, 0.99)),
            "frac_gate_lt_0_01": _format_float(frac_low),
            "frac_gate_gt_0_99": _format_float(frac_high),
            "worst_channel_frac_gate_lt_0_01": _format_float(worst_frac_low),
            "worst_channel_frac_gate_gt_0_99": _format_float(worst_frac_high),
            "dead_channel_count": str(dead_channel_count),
            "input_rms": _format_float(input_rms),
            "output_rms": _format_float(output_rms),
            "output_input_rms_ratio": _format_float(output_input_ratio),
            "a_grad_norm": _format_float(a_grad_norm),
            "b_grad_norm": _format_float(b_grad_norm),
            "a_update_to_param_norm": _format_float(
                _safe_ratio(_tensor_norm(a_after - a_before), _tensor_norm(a_before)),
            ),
            "b_update_to_param_norm": _format_float(
                _safe_ratio(_tensor_norm(b_after - b_before), _tensor_norm(b_before)),
            ),
            "gate_force_fp32": _format_bool(value=module.force_fp32),
            "input_dtype": module.last_input_dtype,
            "gate_math_dtype": module.last_gate_math_dtype,
            "gate_tensor_dtype": module.last_gate_tensor_dtype,
            "output_dtype": module.last_output_dtype,
            "requested_autocast_dtype": "",
            "precision_proof_status": module.last_precision_proof_status,
            "gate_health_status": row_status,
        })
    return rows


def _named_gate_modules(
    model: NonEquivariantVAE,
) -> list[tuple[str, GatedScalarActivation]]:
    modules = cast("Iterable[tuple[object, object]]", model.named_modules())
    return [
        (str(name), module)
        for name, module in modules
        if isinstance(module, GatedScalarActivation)
    ]


def _gate_health_status_from_metrics(  # noqa: PLR0913
    *,
    base_status: str,
    finite_pass: bool,
    max_abs_a: float,
    max_abs_b: float,
    max_saturation_fraction: float,
    worst_channel_saturation_fraction: float,
    dead_channel_count: int,
    num_elements: int,
    num_channels: int,
    output_input_rms_ratio: float,
) -> str:
    has_activation_capture = num_elements > 0
    failure_checks = (
        not finite_pass,
        max_abs_a > GATE_FAILURE_ABS_PARAM,
        max_abs_b > GATE_FAILURE_ABS_PARAM,
        (
            has_activation_capture
            and worst_channel_saturation_fraction >= GATE_FAILURE_SATURATION_FRACTION
        ),
        (
            has_activation_capture
            and dead_channel_count > max(1, int(0.10 * num_channels))
        ),
        (
            has_activation_capture
            and output_input_rms_ratio < GATE_FAILURE_OUTPUT_INPUT_RATIO
        ),
    )
    if any(failure_checks):
        return FAIL_STATUS
    warning_checks = (
        max_abs_a > GATE_WARNING_ABS_PARAM,
        max_abs_b > GATE_WARNING_ABS_PARAM,
        (
            has_activation_capture
            and max_saturation_fraction >= GATE_WARNING_SATURATION_FRACTION
        ),
        has_activation_capture and dead_channel_count > 0,
        (
            has_activation_capture
            and output_input_rms_ratio < GATE_WARNING_OUTPUT_INPUT_RATIO
        ),
    )
    if any(warning_checks):
        return "warn"
    return base_status


def _gate_values(
    input_tensor: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    import torch  # noqa: PLC0415

    if input_tensor.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    return torch.sigmoid(
        (input_tensor * a.reshape(1, -1, 1, 1)) + b.reshape(1, -1, 1, 1),
    )


def _channel_rms(tensor: torch.Tensor) -> torch.Tensor:
    import torch  # noqa: PLC0415

    if tensor.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    if tensor.ndim < MIN_CHANNEL_TENSOR_NDIM:
        return torch.sqrt(torch.mean(tensor.square())).reshape(1)
    reduce_dims = tuple(index for index in range(tensor.ndim) if index != 1)
    return torch.sqrt(torch.mean(tensor.square(), dim=reduce_dims))


def _numerical_delta_payload(  # noqa: PLR0914
    *,
    reference: JsonObject,
    candidate: JsonObject,
) -> JsonObject:
    losses_ref = _required_object(reference, "losses")
    losses_cand = _required_object(candidate, "losses")
    loss_delta = _delta_pair(
        _required_float(losses_ref, "loss"),
        _required_float(losses_cand, "loss"),
    )
    recon_delta = _delta_pair(
        _required_float(losses_ref, "recon_loss"),
        _required_float(losses_cand, "recon_loss"),
    )
    l1_delta = _delta_pair(
        _required_float(losses_ref, "l1_loss"),
        _required_float(losses_cand, "l1_loss"),
    )
    ssim_delta = _delta_pair(
        _required_float(losses_ref, "ssim_loss"),
        _required_float(losses_cand, "ssim_loss"),
    )
    kl_delta = _delta_pair(
        _required_float(losses_ref, "kl_loss"),
        _required_float(losses_cand, "kl_loss"),
    )
    grad_delta = _delta_pair(
        _required_float(reference, "grad_norm"),
        _required_float(candidate, "grad_norm"),
    )
    update_delta = _delta_pair(
        _required_float(reference, "param_update_norm"),
        _required_float(candidate, "param_update_norm"),
    )
    x_hat_min_delta = _delta_pair(
        _required_float(reference, "x_hat_min"),
        _required_float(candidate, "x_hat_min"),
    )
    x_hat_max_delta = _delta_pair(
        _required_float(reference, "x_hat_max"),
        _required_float(candidate, "x_hat_max"),
    )
    mu_mean_delta = _delta_pair(
        _required_float(reference, "mu_mean"),
        _required_float(candidate, "mu_mean"),
    )
    mu_std_delta = _delta_pair(
        _required_float(reference, "mu_std"),
        _required_float(candidate, "mu_std"),
    )
    logvar_mean_delta = _delta_pair(
        _required_float(reference, "logvar_mean"),
        _required_float(candidate, "logvar_mean"),
    )
    logvar_std_delta = _delta_pair(
        _required_float(reference, "logvar_std"),
        _required_float(candidate, "logvar_std"),
    )
    passed = (
        _delta_pass(loss_delta)
        and _delta_pass(recon_delta)
        and _delta_pass(l1_delta)
        and _delta_pass(ssim_delta)
        and kl_delta[1] <= NUMERICAL_KL_REL_THRESHOLD
        and grad_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and update_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and _delta_pass(x_hat_min_delta)
        and _delta_pass(x_hat_max_delta)
        and mu_mean_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and mu_std_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and logvar_mean_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and logvar_std_delta[1] <= NUMERICAL_NORM_REL_THRESHOLD
        and _required_int(reference, "nonfinite_count") == 0
        and _required_int(candidate, "nonfinite_count") == 0
        and _required_int(reference, "logvar_clamp_count")
        == _required_int(candidate, "logvar_clamp_count")
    )
    return {
        "passed": passed,
        "total_loss_abs_delta": loss_delta[0],
        "total_loss_rel_delta": loss_delta[1],
        "recon_loss_abs_delta": recon_delta[0],
        "recon_loss_rel_delta": recon_delta[1],
        "l1_loss_abs_delta": l1_delta[0],
        "l1_loss_rel_delta": l1_delta[1],
        "ssim_loss_abs_delta": ssim_delta[0],
        "ssim_loss_rel_delta": ssim_delta[1],
        "kl_loss_abs_delta": kl_delta[0],
        "kl_loss_rel_delta": kl_delta[1],
        "grad_norm_abs_delta": grad_delta[0],
        "grad_norm_rel_delta": grad_delta[1],
        "param_update_norm_abs_delta": update_delta[0],
        "param_update_norm_rel_delta": update_delta[1],
        "x_hat_min_abs_delta": x_hat_min_delta[0],
        "x_hat_max_abs_delta": x_hat_max_delta[0],
        "mu_mean_abs_delta": mu_mean_delta[0],
        "mu_std_abs_delta": mu_std_delta[0],
        "logvar_mean_abs_delta": logvar_mean_delta[0],
        "logvar_std_abs_delta": logvar_std_delta[0],
        "logvar_clamp_count_delta": abs(
            _required_int(reference, "logvar_clamp_count")
            - _required_int(candidate, "logvar_clamp_count"),
        ),
        "nonfinite_count": (
            _required_int(reference, "nonfinite_count")
            + _required_int(candidate, "nonfinite_count")
        ),
        "amp_step_skipped": False,
    }


def _numerical_row(  # noqa: PLR0913
    *,
    settings: RealDataRuntimePretestSettings,
    row: CsvRow,
    status: str,
    delta_payload: JsonObject,
    batch_index: int,
    reference_row_id: str,
) -> CsvRow:
    supported = status != SKIPPED_UNSUPPORTED
    return {
        "run_name": settings.run_name,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": row["accelerator_mode"],
        "machine_shape": "NvidiaTeslaT4",
        "row_id": row["row_id"],
        "reference_row_id": reference_row_id,
        "candidate_row_id": row["row_id"],
        "batch_index": str(batch_index),
        "precision_policy": row["precision_policy"],
        "torch_compile_enabled": row["torch_compile_enabled"],
        "compile_scope": row["compile_scope"],
        "corruption_strategy": row["corruption_strategy"],
        "total_loss_abs_delta": _payload_float(
            delta_payload,
            "total_loss_abs_delta",
            supported=supported,
        ),
        "total_loss_rel_delta": _payload_float(
            delta_payload,
            "total_loss_rel_delta",
            supported=supported,
        ),
        "recon_loss_abs_delta": _payload_float(
            delta_payload,
            "recon_loss_abs_delta",
            supported=supported,
        ),
        "recon_loss_rel_delta": _payload_float(
            delta_payload,
            "recon_loss_rel_delta",
            supported=supported,
        ),
        "l1_loss_abs_delta": _payload_float(
            delta_payload,
            "l1_loss_abs_delta",
            supported=supported,
        ),
        "l1_loss_rel_delta": _payload_float(
            delta_payload,
            "l1_loss_rel_delta",
            supported=supported,
        ),
        "ssim_loss_abs_delta": _payload_float(
            delta_payload,
            "ssim_loss_abs_delta",
            supported=supported,
        ),
        "ssim_loss_rel_delta": _payload_float(
            delta_payload,
            "ssim_loss_rel_delta",
            supported=supported,
        ),
        "kl_loss_abs_delta": _payload_float(
            delta_payload,
            "kl_loss_abs_delta",
            supported=supported,
        ),
        "kl_loss_rel_delta": _payload_float(
            delta_payload,
            "kl_loss_rel_delta",
            supported=supported,
        ),
        "grad_norm_abs_delta": _payload_float(
            delta_payload,
            "grad_norm_abs_delta",
            supported=supported,
        ),
        "grad_norm_rel_delta": _payload_float(
            delta_payload,
            "grad_norm_rel_delta",
            supported=supported,
        ),
        "param_update_norm_abs_delta": _payload_float(
            delta_payload,
            "param_update_norm_abs_delta",
            supported=supported,
        ),
        "param_update_norm_rel_delta": _payload_float(
            delta_payload,
            "param_update_norm_rel_delta",
            supported=supported,
        ),
        "x_hat_min_abs_delta": _payload_float(
            delta_payload,
            "x_hat_min_abs_delta",
            supported=supported,
        ),
        "x_hat_max_abs_delta": _payload_float(
            delta_payload,
            "x_hat_max_abs_delta",
            supported=supported,
        ),
        "mu_mean_abs_delta": _payload_float(
            delta_payload,
            "mu_mean_abs_delta",
            supported=supported,
        ),
        "mu_std_abs_delta": _payload_float(
            delta_payload,
            "mu_std_abs_delta",
            supported=supported,
        ),
        "logvar_mean_abs_delta": _payload_float(
            delta_payload,
            "logvar_mean_abs_delta",
            supported=supported,
        ),
        "logvar_std_abs_delta": _payload_float(
            delta_payload,
            "logvar_std_abs_delta",
            supported=supported,
        ),
        "logvar_clamp_count_delta": str(
            _required_int(delta_payload, "logvar_clamp_count_delta"),
        )
        if supported
        else "",
        "gate_health_status": status if supported else SKIPPED_UNSUPPORTED,
        "nonfinite_count": str(_required_int(delta_payload, "nonfinite_count"))
        if supported
        else "",
        "amp_step_skipped": _format_bool(
            value=_required_bool(delta_payload, "amp_step_skipped"),
        )
        if supported
        else "",
        "status": status,
        "failure_kind": ""
        if status in {PASS_STATUS, LOCAL_PASS_STATUS}
        else "compile_or_ddp_numerical_pending",
    }


def _corruption_row(  # noqa: PLR0913
    *,
    settings: RealDataRuntimePretestSettings,
    row: CsvRow,
    status: str,
    batch_index: int,
    branchless: JsonObject,
    indexed: JsonObject,
) -> CsvRow:
    supported = status != SKIPPED_UNSUPPORTED
    source = indexed if row["corruption_strategy"] == INDEXED_MASKED else branchless
    reference_row_id = row["row_id"].replace(INDEXED_MASKED, BRANCHLESS_ALL)
    return {
        "run_name": settings.run_name,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": row["accelerator_mode"],
        "machine_shape": "NvidiaTeslaT4",
        "row_id": row["row_id"],
        "reference_row_id": reference_row_id,
        "candidate_row_id": row["row_id"],
        "batch_index": str(batch_index),
        "corruption_version": "spec0001.hed_corruptor.v1",
        "profile_name": "conservative_default",
        "corruption_strategy": row["corruption_strategy"],
        "corruption_view": "train_corrupted_real_data_runtime_pretest",
        "corruption_step": str(batch_index),
        "split": "train",
        "semantic_sample_key_hash": _required_str(source, "semantic_sample_key_hash")
        if supported
        else "",
        "binary_sample_id_hash": _required_str(source, "sample_id_hash")
        if supported
        else "",
        "rank": "0",
        "world_size": row["world_size"],
        "applied_mask_hash": _required_str(source, "applied_mask_hash")
        if supported
        else "",
        "stain_param_hash": _required_str(source, "stain_param_hash")
        if supported
        else "",
        "noise_std_hash": _required_str(source, "noise_std_hash") if supported else "",
        "noise_field_hash": _required_str(source, "gaussian_only_hash")
        if supported
        else "",
        "clean_sample_unchanged_count": str(
            _required_int(source, "clean_sample_unchanged_count"),
        )
        if supported
        else "",
        "clean_validation_rng_advanced": "",
        "status": status,
        "failure_kind": ""
        if status in {PASS_STATUS, LOCAL_PASS_STATUS}
        else "candidate_specific_corruption_pending",
    }


def _delta_pair(reference: float, candidate: float) -> tuple[float, float]:
    absolute = abs(candidate - reference)
    relative = absolute / max(abs(reference), 1.0e-8)
    return (absolute, relative)


def _delta_pass(delta: tuple[float, float]) -> bool:
    return delta[0] <= NUMERICAL_ABS_THRESHOLD or delta[1] <= NUMERICAL_REL_THRESHOLD


def _payload_float(payload: JsonObject, key: str, *, supported: bool) -> str:
    return _format_float(_required_float(payload, key)) if supported else ""


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _metadata_hash(values: Sequence[JsonObject]) -> str:
    return _hash_text(
        json.dumps(values, sort_keys=True, separators=(",", ":"), default=str),
    )


def _clean_sample_unchanged_count(
    *,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    applied: Sequence[bool],
) -> int:
    import torch  # noqa: PLC0415

    count = 0
    for index, was_applied in enumerate(applied):
        if not was_applied and bool(torch.equal(clean[index], corrupted[index])):
            count += 1
    return count


def _tensor_mean(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.mean().item())


def _tensor_std(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.std(unbiased=False).item())


def _tensor_quantile(tensor: torch.Tensor, quantile: float) -> float:
    if tensor.numel() == 0:
        return 0.0
    values = _gate_quantile_values(tensor)
    return float(torch.quantile(values, quantile).item())


def _gate_quantile_values(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=tensor.device)
    if tensor.numel() <= MAX_GATE_QUANTILE_ELEMENTS:
        return tensor.flatten()
    values = tensor if tensor.is_contiguous() else tensor.contiguous()
    flattened = values.view(-1)
    sample_count = min(MAX_GATE_QUANTILE_ELEMENTS, int(flattened.numel()))
    indices = torch.linspace(
        0,
        int(flattened.numel()) - 1,
        steps=sample_count,
        device=flattened.device,
        dtype=torch.float64,
    ).round()
    return flattened.index_select(0, indices.to(dtype=torch.long))


def _tensor_fraction(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.to(dtype=torch.float32).mean().item())


def _worst_channel_fraction(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    if mask.ndim < MIN_CHANNEL_TENSOR_NDIM:
        return _tensor_fraction(mask)
    reduce_dims = tuple(index for index in range(mask.ndim) if index != 1)
    per_channel = mask.to(dtype=torch.float32).mean(dim=reduce_dims)
    return float(torch.amax(per_channel).item())


def _tensor_rms(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return math.sqrt(float(tensor.to(dtype=torch.float32).square().mean().item()))


def _tensor_norm(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return math.sqrt(float(tensor.to(dtype=torch.float32).square().sum().item()))


def _optional_tensor_norm(tensor: torch.Tensor | None) -> float:
    return 0.0 if tensor is None else _tensor_norm(tensor)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(abs(denominator), 1.0e-8)


def _rows_with_linked_evidence(
    *,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    numerical_rows = _csv_rows_from_payload(
        _required_object(linked_evidence, "paired_numerical"),
        "rows",
    )
    corruption_rows = _csv_rows_from_payload(
        _required_object(linked_evidence, "corruption_equivalence"),
        "rows",
    )
    updated: list[CsvRow] = []
    for row in rows:
        new_row = dict(row)
        if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}:
            new_row["gate_health_status"] = _row_gate_status(
                linked_evidence=linked_evidence,
                row=new_row,
                default=SKIPPED_UNSUPPORTED,
            )
            if new_row["gate_health_status"] in {PASS_STATUS, LOCAL_PASS_STATUS}:
                new_row["gate_health_warning_count"] = "0"
            new_row["numerical_check_status"] = _row_status_for_row_id(
                rows=numerical_rows,
                row_id=row["row_id"],
                default=SKIPPED_UNSUPPORTED,
            )
            data_wait_fraction = _dataloader_wait_fraction_for_row(
                linked_evidence=linked_evidence,
                row=new_row,
            )
            if data_wait_fraction:
                new_row["data_wait_fraction_p95"] = data_wait_fraction
            if row["compile_scope"] == COMPILE_NONE:
                new_row["graph_break_count"] = "0"
                new_row["recompile_count"] = "0"
            if _row_linked_evidence_pass(
                row=new_row,
                data_proof=data_proof,
                linked_evidence=linked_evidence,
                numerical_rows=numerical_rows,
                corruption_rows=corruption_rows,
            ):
                new_row["status"] = PASS_STATUS
                new_row["failure_kind"] = ""
                new_row["failure_message_hash"] = ""
            else:
                new_row["status"] = INELIGIBLE_STATUS
                new_row["failure_kind"] = _row_ineligibility_reason(
                    row=new_row,
                    data_proof=data_proof,
                    linked_evidence=linked_evidence,
                    numerical_rows=numerical_rows,
                    corruption_rows=corruption_rows,
                )
                new_row["failure_message_hash"] = _hash_text(new_row["failure_kind"])
        updated.append(new_row)
    return updated


def _row_status_for_row_id(
    *,
    rows: Sequence[Mapping[str, str]],
    row_id: str,
    default: str,
) -> str:
    matches = [
        row.get("status", default) for row in rows if row.get("row_id") == row_id
    ]
    return _combined_linked_row_status(matches) if matches else default


def _row_gate_status(
    *,
    linked_evidence: JsonObject,
    row: CsvRow,
    default: str,
) -> str:
    gate_health = _required_object(linked_evidence, "gate_health")
    row_statuses = gate_health.get("row_statuses")
    if isinstance(row_statuses, list):
        for item in row_statuses:
            if not isinstance(item, dict):
                continue
            row_status = cast("JsonObject", item)
            if row_status.get("row_id") == row["row_id"] or _row_status_matches(
                row_status=row_status,
                row=row,
            ):
                return _required_str(row_status, "status")
    return default


def _row_status_matches(*, row_status: JsonObject, row: CsvRow) -> bool:
    try:
        return (
            _required_str(row_status, "accelerator_mode") == row["accelerator_mode"]
            and str(_required_int(row_status, "world_size")) == row["world_size"]
            and str(_required_int(row_status, "per_device_batch_size"))
            == row["per_device_batch_size"]
            and _required_str(row_status, "precision_policy") == row["precision_policy"]
            and _required_str(row_status, "compile_scope") == row["compile_scope"]
        )
    except (TypeError, ValueError):
        return False


def _dataloader_wait_fraction_for_row(
    *,
    linked_evidence: JsonObject,
    row: CsvRow,
) -> str:
    dataloader_rows = _dataloader_rows_for_runtime_row(
        linked_evidence=linked_evidence,
        row=row,
    )
    values = [
        _csv_float_or_inf(dataloader_row, "data_wait_fraction_p95")
        for dataloader_row in dataloader_rows
        if dataloader_row["status"] in {PASS_STATUS, LOCAL_PASS_STATUS}
    ]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return ""
    return _format_float(max(finite_values))


def _dataloader_rows_for_runtime_row(
    *,
    linked_evidence: JsonObject,
    row: CsvRow,
) -> list[dict[str, str]]:
    dataloader = _required_object(linked_evidence, "dataloader_throughput")
    dataloader_rows = _csv_rows_from_payload(dataloader, "rows")
    return [
        dataloader_row
        for dataloader_row in dataloader_rows
        if dataloader_row["accelerator_mode"] == row["accelerator_mode"]
        and dataloader_row["world_size"] == row["world_size"]
        and dataloader_row["batch_size"] == row["per_device_batch_size"]
    ]


def _dataloader_evidence_pass_for_row(
    *,
    linked_evidence: JsonObject,
    row: CsvRow,
) -> bool:
    rows = _dataloader_rows_for_runtime_row(linked_evidence=linked_evidence, row=row)
    split_statuses = {
        dataloader_row["split"]: dataloader_row["status"] for dataloader_row in rows
    }
    return (
        split_statuses.get("train") == PASS_STATUS
        and split_statuses.get("validation") == PASS_STATUS
        and all(
            _dataloader_row_threshold_pass(row=dataloader_row)
            for dataloader_row in rows
        )
    )


def _dataloader_row_threshold_pass(*, row: Mapping[str, str]) -> bool:
    wait_fraction = _csv_float_or_inf(row, "data_wait_fraction_p95")
    loader_samples_sec = _csv_float_or_inf(row, "loader_samples_sec")
    trainer_samples_sec = _csv_float_or_inf(row, "trainer_samples_sec")
    return (
        math.isfinite(wait_fraction)
        and wait_fraction <= MAX_DATA_WAIT_FRACTION
        and math.isfinite(loader_samples_sec)
        and math.isfinite(trainer_samples_sec)
        and trainer_samples_sec > 0.0
        and loader_samples_sec
        >= MIN_LOADER_TRAINER_THROUGHPUT_RATIO * trainer_samples_sec
    )


def _canonical_data_proof_pass(data_proof: JsonObject) -> bool:
    return (
        _required_str(data_proof, "identity_status") == PASS_STATUS
        and _required_str(data_proof, "row_count_status") == PASS_STATUS
        and _required_str(data_proof, "crc_validation_status") == PASS_STATUS
        and _required_str(data_proof, "window_status") == PASS_STATUS
        and _required_str(data_proof, "clean_validation_dataloader_status")
        == PASS_STATUS
    )


def _compile_evidence_pass_for_row(
    *,
    row: CsvRow,
    linked_evidence: JsonObject,
) -> bool:
    if row["compile_scope"] == COMPILE_NONE:
        return (
            _csv_int_or_zero(row, "graph_break_count") == 0
            and _csv_int_or_zero(row, "recompile_count") == 0
        )
    # COMPILE_STEP (whole-step compile) is screened on the same relationship as
    # COMPILE_MODEL_FORWARD: a PASS compile-settle lane plus zero post-settle graph
    # breaks/recompiles. Both are measured but kept ineligible until full settle-path
    # coverage lands (see ``_compile_settle_proof``).
    if row["compile_scope"] not in {COMPILE_MODEL_FORWARD, COMPILE_STEP}:
        return False
    compile_settle = _required_object(linked_evidence, "compile_settle")
    return (
        _required_str(compile_settle, "status") == PASS_STATUS
        and _csv_int_or_zero(row, "graph_break_count") == 0
        and _csv_int_or_zero(row, "recompile_count") == 0
    )


def _row_linked_evidence_pass(
    *,
    row: CsvRow,
    data_proof: JsonObject,
    linked_evidence: JsonObject,
    numerical_rows: Sequence[Mapping[str, str]],
    corruption_rows: Sequence[Mapping[str, str]],
) -> bool:
    return (
        _canonical_data_proof_pass(data_proof)
        and _linked_status(linked_evidence, "ddp_launch") == PASS_STATUS
        and _compile_evidence_pass_for_row(row=row, linked_evidence=linked_evidence)
        and _dataloader_evidence_pass_for_row(
            linked_evidence=linked_evidence,
            row=row,
        )
        and _row_status_for_row_id(
            rows=numerical_rows,
            row_id=row["row_id"],
            default=SKIPPED_UNSUPPORTED,
        )
        == PASS_STATUS
        and _row_status_for_row_id(
            rows=corruption_rows,
            row_id=row["row_id"],
            default=SKIPPED_UNSUPPORTED,
        )
        == PASS_STATUS
        and _row_gate_status(
            linked_evidence=linked_evidence,
            row=row,
            default=SKIPPED_UNSUPPORTED,
        )
        == PASS_STATUS
    )


def _row_ineligibility_reason(
    *,
    row: CsvRow,
    data_proof: JsonObject,
    linked_evidence: JsonObject,
    numerical_rows: Sequence[Mapping[str, str]],
    corruption_rows: Sequence[Mapping[str, str]],
) -> str:
    checks = (
        (
            _required_str(data_proof, "identity_status") != PASS_STATUS,
            "canonical_real_identity_evidence_not_pass",
        ),
        (
            _required_str(data_proof, "row_count_status") != PASS_STATUS,
            "canonical_real_row_count_evidence_not_pass",
        ),
        (
            _required_str(data_proof, "crc_validation_status") != PASS_STATUS,
            "canonical_real_crc_evidence_not_pass",
        ),
        (
            _required_str(data_proof, "window_status") != PASS_STATUS,
            "canonical_real_window_evidence_not_pass",
        ),
        (
            _required_str(data_proof, "clean_validation_dataloader_status")
            != PASS_STATUS,
            "canonical_clean_validation_loader_evidence_not_pass",
        ),
        (
            _linked_status(linked_evidence, "ddp_launch") != PASS_STATUS,
            "ddp_launch_evidence_not_canonical_pass",
        ),
        (
            not _compile_evidence_pass_for_row(
                row=row,
                linked_evidence=linked_evidence,
            ),
            "compile_settle_or_dynamo_evidence_not_row_pass",
        ),
        (
            not _dataloader_evidence_pass_for_row(
                linked_evidence=linked_evidence,
                row=row,
            ),
            "dataloader_throughput_evidence_not_row_pass",
        ),
        (
            _row_status_for_row_id(
                rows=numerical_rows,
                row_id=row["row_id"],
                default=SKIPPED_UNSUPPORTED,
            )
            != PASS_STATUS,
            "paired_numerical_evidence_not_row_pass",
        ),
        (
            _row_status_for_row_id(
                rows=corruption_rows,
                row_id=row["row_id"],
                default=SKIPPED_UNSUPPORTED,
            )
            != PASS_STATUS,
            "corruption_equivalence_evidence_not_row_pass",
        ),
        (
            _row_gate_status(
                linked_evidence=linked_evidence,
                row=row,
                default=SKIPPED_UNSUPPORTED,
            )
            != PASS_STATUS,
            "gate_health_evidence_not_row_pass",
        ),
    )
    for failed, reason in checks:
        if failed:
            return reason
    return "linked_safety_evidence_pending"


def _manifest_payload(  # noqa: PLR0913
    *,
    request: RealDataRuntimePretestRequest,
    resolved: ResolvedConfig,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
    linked_evidence: JsonObject,
    phase_timings: JsonObject,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": REAL_DATA_PRETEST_SCHEMA_VERSION,
            "status": "pretest_manifest_ready",
            "status_scope": REAL_DATA_PRETEST_SCOPE,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": False,
            "writes_selected_runtime": False,
            "blocked_claims": settings.blocked_claims,
            "config_path": str(request.config_path),
            "config_sha256": resolved.invoked_config_hash,
            "effective_config_sha256": resolved.effective_config_hash,
            "dataset_slug": settings.dataset_slug,
            "data_root": settings.data_root,
            "train_windows": [
                _window_payload(window) for window in settings.train_windows
            ],
            "validation_windows": [
                _window_payload(window) for window in settings.validation_windows
            ],
            "real_data_identity_proof_status": _required_str(
                data_proof,
                "identity_status",
            ),
            "file_hashes": _required_object_list(data_proof, "file_hashes"),
            "row_count_proof_status": _required_str(data_proof, "row_count_status"),
            "wsi_count_proof_status": _required_str(data_proof, "wsi_count_status"),
            "crc_validation_status": _required_str(
                data_proof,
                "crc_validation_status",
            ),
            "cache_warmup_policy": (
                "sha256_crc_window_audit_then_clean_validation_dataloader"
                if _required_str(data_proof, "status") == PASS_STATUS
                else "not_run"
            ),
            "train_windows_exercised": _proof_status_exercised(
                _required_str(data_proof, "window_status"),
            ),
            "validation_windows_exercised": (
                _proof_status_exercised(_required_str(data_proof, "window_status"))
                and _proof_status_exercised(
                    _required_str(data_proof, "clean_validation_dataloader_status"),
                )
            ),
            "clean_validation_dataloader_proof": _required_object(
                data_proof,
                "clean_validation_dataloader",
            ),
            "compile_settle_proof": _required_object(
                linked_evidence,
                "compile_settle",
            ),
            "ddp_launch_proof": _required_object(linked_evidence, "ddp_launch"),
            "dataloader_throughput_proof": _required_object(
                linked_evidence,
                "dataloader_throughput",
            ),
            "paired_numerical_proof": _required_object(
                linked_evidence,
                "paired_numerical",
            ),
            "corruption_equivalence_proof": _required_object(
                linked_evidence,
                "corruption_equivalence",
            ),
            "gate_health_proof": _required_object(linked_evidence, "gate_health"),
            "linked_evidence_status": _required_str(linked_evidence, "status"),
            "real_data_proof": data_proof,
            "phase_timings": phase_timings,
            "timed_rows_eligible": any(row["status"] == PASS_STATUS for row in rows),
            "seeded_candidates": [
                _candidate_payload(candidate)
                for candidate in settings.seeded_candidates
            ],
            "artifact_allowlist": [
                MANIFEST_FILENAME,
                RUNTIME_PROOF_FILENAME,
                RUNTIME_MATRIX_FILENAME,
                DATALOADER_MATRIX_FILENAME,
                NUMERICAL_CHECKS_FILENAME,
                CORRUPTION_CHECKS_FILENAME,
                f"metrics/{GATE_HEALTH_FILENAME}",
                GATE_HEALTH_SUMMARY_FILENAME,
                RECOMMENDATIONS_FILENAME,
                PHASE_TIMINGS_FILENAME,
            ],
            "selected_runtime_written": False,
        },
    )


def _runtime_proof_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
    linked_evidence: JsonObject,
    phase_timings: JsonObject,
) -> JsonObject:
    paired_numerical = _required_object(linked_evidence, "paired_numerical")
    corruption_equivalence = _required_object(
        linked_evidence,
        "corruption_equivalence",
    )
    return cast(
        "JsonObject",
        {
            "schema_version": REAL_DATA_PRETEST_SCHEMA_VERSION,
            "status": _overall_status(rows),
            "status_scope": REAL_DATA_PRETEST_SCOPE,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": False,
            "blocked_claims": settings.blocked_claims,
            "dataset_slug": settings.dataset_slug,
            "machine_shape": "NvidiaTeslaT4",
            "accelerator_modes_configured": [SINGLE_VISIBLE_T4, DUAL_T4_DDP],
            "accelerator_modes_with_timing": sorted(
                {
                    row["accelerator_mode"]
                    for row in rows
                    if row["status"] in {PASS_STATUS, INELIGIBLE_STATUS}
                },
            ),
            "row_count": len(rows),
            "eligible_pass_row_count": sum(
                1 for row in rows if row["status"] == PASS_STATUS
            ),
            "timed_ineligible_row_count": sum(
                1 for row in rows if row["status"] == INELIGIBLE_STATUS
            ),
            "skipped_unsupported_row_count": sum(
                1 for row in rows if row["status"] == SKIPPED_UNSUPPORTED
            ),
            "wrong_accelerator_row_count": sum(
                1 for row in rows if row["status"] == WRONG_ACCELERATOR
            ),
            "real_data_identity_proof_status": _required_str(
                data_proof,
                "identity_status",
            ),
            "crc_validation_status": _required_str(
                data_proof,
                "crc_validation_status",
            ),
            "window_status": _required_str(data_proof, "window_status"),
            "clean_validation_dataloader_status": _required_str(
                data_proof,
                "clean_validation_dataloader_status",
            ),
            "linked_evidence_status": _required_str(linked_evidence, "status"),
            "dataloader_throughput_status": _linked_status(
                linked_evidence,
                "dataloader_throughput",
            ),
            "paired_numerical_status": _linked_status(
                linked_evidence,
                "paired_numerical",
            ),
            "paired_numerical_candidate_evidence_count": _optional_int(
                paired_numerical,
                "candidate_evidence_count",
            )
            or 0,
            "paired_numerical_failed_candidate_evidence_count": _optional_int(
                paired_numerical,
                "failed_candidate_evidence_count",
            )
            or 0,
            "corruption_equivalence_status": _linked_status(
                linked_evidence,
                "corruption_equivalence",
            ),
            "corruption_equivalence_candidate_evidence_count": _optional_int(
                corruption_equivalence,
                "candidate_evidence_count",
            )
            or 0,
            "corruption_equivalence_failed_candidate_evidence_count": _optional_int(
                corruption_equivalence,
                "failed_candidate_evidence_count",
            )
            or 0,
            "gate_health_status": _linked_status(linked_evidence, "gate_health"),
            "ddp_launch_status": _linked_status(linked_evidence, "ddp_launch"),
            "compile_settle_policy": {
                "compile_settle_steps": settings.compile_settle_steps,
                "counter_source": "torch._dynamo.utils.counters_with_reset_per_row",
                "implemented_in_this_runner": True,
                "implemented_compile_scopes": [COMPILE_MODEL_FORWARD, COMPILE_STEP],
                "contract_proof_available": True,
                "status": _linked_status(linked_evidence, "compile_settle"),
                "proof": _required_object(linked_evidence, "compile_settle"),
            },
            "selection_ready": False,
            "selected_runtime_written": False,
            "phase_timings": phase_timings,
            "evidence_gate": _runtime_evidence_gate(
                rows=rows,
                data_proof=data_proof,
                linked_evidence=linked_evidence,
            ),
        },
    )


def _recommendations_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
) -> JsonObject:
    return {
        "schema_version": REAL_DATA_PRETEST_SCHEMA_VERSION,
        "status": _overall_status(rows),
        "status_scope": REAL_DATA_PRETEST_SCOPE,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": False,
        "writes_selected_runtime": False,
        "blocked_claims": settings.blocked_claims,
        "recommendations": [
            _recommendation(row=row, rank=index + 1)
            for index, row in enumerate(sorted(rows, key=_recommendation_sort_key))
        ],
        "selection_policy": (
            "This capped pretest may carry rows forward for fuller benchmark "
            "implementation, but it never writes selected_runtime.json."
        ),
    }


def _recommendation(*, row: CsvRow, rank: int) -> JsonObject:
    status = row["status"]
    if status == PASS_STATUS:
        recommendation = "eligible_after_linked_evidence"
    elif status == INELIGIBLE_STATUS:
        recommendation = "measured_but_linked_evidence_pending_do_not_select"
    elif status == SKIPPED_UNSUPPORTED:
        recommendation = "implementation_pending"
    elif status == WRONG_ACCELERATOR:
        recommendation = "rerun_on_correct_accelerator"
    else:
        recommendation = "inspect_failure"
    return {
        "recommendation_rank": rank,
        "row_id": row["row_id"],
        "status": status,
        "recommendation": recommendation,
        "estimated_epoch_minutes": "",
        "samples_sec": row["samples_sec"],
        "steady_step_ms_p95": row["steady_step_ms_p95"],
    }


def _runtime_evidence_gate(
    *,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
    linked_evidence: JsonObject,
) -> str:
    if any(row["status"] == PASS_STATUS for row in rows):
        return (
            "At least one capped real-data timing row has passing "
            "candidate-specific linked evidence. Rows remain non-promotable "
            "because this capped pretest must not write selected_runtime.json; "
            "a later selected-runtime benchmark must consume these linked "
            "artifacts explicitly."
        )
    if (
        _required_str(data_proof, "identity_status") == PASS_STATUS
        and _required_str(data_proof, "crc_validation_status") == PASS_STATUS
        and _required_str(data_proof, "window_status") == PASS_STATUS
        and _required_str(data_proof, "clean_validation_dataloader_status")
        == PASS_STATUS
        and _required_str(linked_evidence, "status") == PASS_STATUS
    ):
        return (
            "All required capped real-data pretest evidence lanes passed. "
            "Rows remain non-promotable because this capped pretest must not "
            "write selected_runtime.json; a later selected-runtime benchmark "
            "must consume these linked artifacts explicitly."
        )
    return (
        "Timing rows remain ineligible until real-data identity, CRC, "
        "validation-window, real dataloader throughput, paired numerical, "
        "corruption, gate-health, DDP, and compile-settle evidence passes."
    )


def _schema_dataloader_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    clean_validation = _required_object(data_proof, "clean_validation_dataloader")
    throughput = _required_object(linked_evidence, "dataloader_throughput")
    measured_rows = _csv_rows_from_payload(throughput, "rows")
    if measured_rows:
        return cast("list[CsvRow]", measured_rows)
    persistent_workers = _format_bool(value=DEFAULT_DATALOADER_PERSISTENT_WORKERS)
    non_blocking_h2d = _format_bool(value=DEFAULT_DATALOADER_NON_BLOCKING_H2D)
    rows: list[CsvRow] = []
    for split in ("train", "validation"):
        row = {
            "run_name": settings.run_name,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": "false",
            "accelerator_mode": SINGLE_VISIBLE_T4,
            "machine_shape": "NvidiaTeslaT4",
            "world_size": "1",
            "rank": "0",
            "split": split,
            "num_workers": str(DEFAULT_DATALOADER_NUM_WORKERS),
            "prefetch_factor": DEFAULT_DATALOADER_PREFETCH_FACTOR,
            "pin_memory": _format_bool(value=DEFAULT_DATALOADER_PIN_MEMORY),
            "persistent_workers": persistent_workers,
            "non_blocking_h2d": non_blocking_h2d,
            "batch_size": "",
            "batches_measured": "0",
            "batch_fetch_ms_p50": "",
            "batch_fetch_ms_p95": "",
            "h2d_ms_p50": "",
            "h2d_ms_p95": "",
            "loader_samples_sec": "",
            "trainer_samples_sec": "",
            "data_wait_fraction_p50": "",
            "data_wait_fraction_p95": "",
            "rank_sample_count": "",
            "dropped_sample_count": "0",
            "status": SKIPPED_UNSUPPORTED,
            "failure_kind": "dataloader_grid_measurement_pending",
        }
        if (
            split == "validation"
            and _required_str(clean_validation, "status") == PASS_STATUS
        ):
            row.update({
                "batch_size": str(_required_int(clean_validation, "batch_size")),
                "batches_measured": str(
                    _required_int(clean_validation, "batches_seen"),
                ),
                "batch_fetch_ms_p50": _required_str(
                    clean_validation,
                    "batch_fetch_ms_p50",
                ),
                "batch_fetch_ms_p95": _required_str(
                    clean_validation,
                    "batch_fetch_ms_p95",
                ),
                "loader_samples_sec": _required_str(
                    clean_validation,
                    "loader_samples_sec",
                ),
                "rank_sample_count": str(
                    _required_int(clean_validation, "sample_count"),
                ),
                "status": INELIGIBLE_STATUS,
                "failure_kind": ("clean_validation_path_pass_throughput_grid_pending"),
            })
        rows.append(row)
    return rows


def _schema_numerical_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    numerical = _required_object(linked_evidence, "paired_numerical")
    measured_rows = _csv_rows_from_payload(numerical, "rows")
    if measured_rows:
        return cast("list[CsvRow]", measured_rows)
    fallback_status = _required_str(numerical, "status")
    fallback_failure_kind = _optional_str(numerical, "failure_kind") or (
        "paired_numerical_checks_pending"
    )
    return [
        {
            "run_name": settings.run_name,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": "false",
            "accelerator_mode": row["accelerator_mode"],
            "machine_shape": "NvidiaTeslaT4",
            "row_id": row["row_id"],
            "reference_row_id": "",
            "candidate_row_id": row["row_id"],
            "batch_index": "0",
            "precision_policy": row["precision_policy"],
            "torch_compile_enabled": row["torch_compile_enabled"],
            "compile_scope": row["compile_scope"],
            "corruption_strategy": row["corruption_strategy"],
            "total_loss_abs_delta": "",
            "total_loss_rel_delta": "",
            "recon_loss_abs_delta": "",
            "recon_loss_rel_delta": "",
            "l1_loss_abs_delta": "",
            "l1_loss_rel_delta": "",
            "ssim_loss_abs_delta": "",
            "ssim_loss_rel_delta": "",
            "kl_loss_abs_delta": "",
            "kl_loss_rel_delta": "",
            "grad_norm_abs_delta": "",
            "grad_norm_rel_delta": "",
            "param_update_norm_abs_delta": "",
            "param_update_norm_rel_delta": "",
            "x_hat_min_abs_delta": "",
            "x_hat_max_abs_delta": "",
            "mu_mean_abs_delta": "",
            "mu_std_abs_delta": "",
            "logvar_mean_abs_delta": "",
            "logvar_std_abs_delta": "",
            "logvar_clamp_count_delta": "",
            "gate_health_status": row["gate_health_status"],
            "nonfinite_count": "",
            "amp_step_skipped": "",
            "status": fallback_status,
            "failure_kind": fallback_failure_kind,
        }
        for row in rows
    ]


def _schema_corruption_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    corruption = _required_object(linked_evidence, "corruption_equivalence")
    measured_rows = _csv_rows_from_payload(corruption, "rows")
    if measured_rows:
        return cast("list[CsvRow]", measured_rows)
    fallback_status = _required_str(corruption, "status")
    fallback_failure_kind = _optional_str(corruption, "failure_kind") or (
        "corruption_equivalence_checks_pending"
    )
    return [
        {
            "run_name": settings.run_name,
            "benchmark_kind": REAL_DATA_PRETEST_KIND,
            "benchmark_source": REAL_DATA_PRETEST_SOURCE,
            "full_run_eligible": "false",
            "accelerator_mode": row["accelerator_mode"],
            "machine_shape": "NvidiaTeslaT4",
            "row_id": row["row_id"],
            "reference_row_id": "",
            "candidate_row_id": row["row_id"],
            "batch_index": "0",
            "corruption_version": "spec0001.hed_corruptor.v1",
            "profile_name": "conservative_default",
            "corruption_strategy": row["corruption_strategy"],
            "corruption_view": "train_corrupted_real_data_runtime_pretest",
            "corruption_step": "0",
            "split": "train",
            "semantic_sample_key_hash": "",
            "binary_sample_id_hash": "",
            "rank": "0",
            "world_size": row["world_size"],
            "applied_mask_hash": "",
            "stain_param_hash": "",
            "noise_std_hash": "",
            "noise_field_hash": "",
            "clean_sample_unchanged_count": "",
            "clean_validation_rng_advanced": "",
            "status": fallback_status,
            "failure_kind": fallback_failure_kind,
        }
        for row in rows
    ]


def _gate_health_summary_payload(*, linked_evidence: JsonObject) -> JsonObject:
    gate_health = _required_object(linked_evidence, "gate_health")
    rows = _csv_rows_from_payload(gate_health, "rows")
    status = _required_str(gate_health, "status")
    if rows:
        failing_modules = [
            row["module"]
            for row in rows
            if row["gate_health_status"] not in {PASS_STATUS, LOCAL_PASS_STATUS}
        ]
        warning_modules = [
            row["module"] for row in rows if row["gate_health_status"] == "warn"
        ]
        return cast(
            "JsonObject",
            {
                "status": status,
                "benchmark_kind": REAL_DATA_PRETEST_KIND,
                "benchmark_source": REAL_DATA_PRETEST_SOURCE,
                "overall_status": status,
                "full_run_eligible": False,
                "logged_intervals": 1,
                "module_count": len(rows),
                "nonfinite_count": _required_int(gate_health, "nonfinite_count"),
                "failing_modules": failing_modules,
                "warning_modules": warning_modules,
                "notes": _required_str(gate_health, "notes"),
            },
        )
    return {
        "status": status,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "overall_status": status,
        "full_run_eligible": False,
        "logged_intervals": 0,
        "module_count": 0,
        "nonfinite_count": None,
        "failing_modules": [],
        "warning_modules": [],
        "notes": _required_str(gate_health, "notes"),
    }


def _gate_health_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    linked_evidence: JsonObject,
) -> list[CsvRow]:
    del settings
    return cast(
        "list[CsvRow]",
        _csv_rows_from_payload(
            _required_object(linked_evidence, "gate_health"),
            "rows",
        ),
    )


def _child_failure_payload(  # noqa: PLR0913
    *,
    settings: RealDataRuntimePretestSettings,
    row_spec: RowSpec,
    status: str,
    failure_kind: str,
    failure_message: str,
    accelerator: JsonObject,
) -> JsonObject:
    return {
        "row_id": row_spec.row_id,
        "status": status,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "full_run_eligible": False,
        "blocked_claims": settings.blocked_claims,
        "failure_kind": failure_kind,
        "failure_message": failure_message,
        "accelerator": accelerator,
    }


def _write_child_payload(output_dir: Path, row_id: str, payload: JsonObject) -> None:
    write_json(output_dir / "benchmark" / "child_rows" / f"{row_id}.json", payload)


def _accelerator_observation() -> JsonObject:
    import torch  # noqa: PLC0415

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return cast(
        "JsonObject",
        {
            "visible_device_count": device_count,
            "cuda_device_count": device_count,
            "gpu_names": names,
            "cuda_available": cuda_available,
        },
    )


def _accelerator_failure(
    *,
    row_spec: RowSpec,
    accelerator: JsonObject,
) -> tuple[str, str, str] | None:
    device_count = _required_int(accelerator, "cuda_device_count")
    gpu_names = _required_str_list(accelerator, "gpu_names")
    if row_spec.accelerator_mode == SINGLE_VISIBLE_T4:
        if device_count != 1 or not _all_t4(gpu_names):
            return (
                WRONG_ACCELERATOR,
                "wrong_accelerator",
                f"expected one visible T4, got count={device_count}, names={gpu_names}",
            )
        return None
    if device_count != DUAL_T4_DEVICE_COUNT or not _all_t4(gpu_names):
        return (
            WRONG_ACCELERATOR,
            "wrong_accelerator",
            (
                "expected two visible T4 devices, got "
                f"count={device_count}, names={gpu_names}"
            ),
        )
    return None


def _all_t4(gpu_names: Sequence[str]) -> bool:
    return bool(gpu_names) and all("T4" in name for name in gpu_names)


def _window_indices(windows: Sequence[WindowSpec]) -> list[int]:
    indices: list[int] = []
    for window in windows:
        indices.extend(range(window.start_row, window.stop_row))
    return indices


def _window_payload(window: WindowSpec) -> JsonObject:
    return {
        "name": window.name,
        "start_row": window.start_row,
        "stop_row": window.stop_row,
        "patch_count": window.patch_count,
    }


def _candidate_payload(candidate: SeededCandidate) -> JsonObject:
    return {
        "accelerator_mode": candidate.accelerator_mode,
        "per_device_batch_size": candidate.per_device_batch_size,
        "synthetic_v4_rank": candidate.synthetic_v4_rank,
        "synthetic_v4_row_id": candidate.synthetic_v4_row_id,
        "candidate_role": candidate.candidate_role,
    }


def _window_specs(payload: JsonObject, key: str) -> tuple[WindowSpec, ...]:
    raw_windows = _required_list(payload, key)
    windows: list[WindowSpec] = []
    for raw_window in raw_windows:
        if not isinstance(raw_window, dict):
            message = f"{key} entries must be objects"
            raise TypeError(message)
        window = cast("JsonObject", raw_window)
        windows.append(
            WindowSpec(
                name=_required_str(window, "name"),
                start_row=_required_int(window, "start_row"),
                patch_count=_required_int(window, "patch_count"),
            ),
        )
    return tuple(windows)


def _seeded_candidates(payload: JsonObject) -> tuple[SeededCandidate, ...]:
    raw_candidates = _required_list(payload, "seeded_candidates")
    candidates: list[SeededCandidate] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            message = "seeded_candidates entries must be objects"
            raise TypeError(message)
        candidate = cast("JsonObject", raw_candidate)
        candidates.append(
            SeededCandidate(
                accelerator_mode=_required_str(candidate, "accelerator_mode"),
                per_device_batch_size=_required_int(
                    candidate,
                    "per_device_batch_size",
                ),
                synthetic_v4_rank=_optional_int(candidate, "synthetic_v4_rank"),
                synthetic_v4_row_id=_optional_str(candidate, "synthetic_v4_row_id"),
                candidate_role=_required_str(candidate, "candidate_role"),
            ),
        )
    return tuple(candidates)


def _channels_from_model(model: JsonObject) -> int:
    input_shape = _int_tuple(model, "input_shape")
    if len(input_shape) != MODEL_INPUT_SHAPE_NDIM:
        message = "model.input_shape must have four dimensions"
        raise ValueError(message)
    return input_shape[1]


def _image_size_from_model(model: JsonObject) -> int:
    input_shape = _int_tuple(model, "input_shape")
    if len(input_shape) != MODEL_INPUT_SHAPE_NDIM:
        message = "model.input_shape must have four dimensions"
        raise ValueError(message)
    if input_shape[2] != input_shape[3]:
        message = "model.input_shape must be square for patch benchmark"
        raise ValueError(message)
    return input_shape[2]


def _str_tuple(payload: JsonObject, key: str) -> tuple[str, ...]:
    values = _required_list(payload, key)
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str):
            message = f"{key} values must be strings"
            raise TypeError(message)
        parsed.append(value)
    return tuple(parsed)


def _int_tuple(payload: JsonObject, key: str) -> tuple[int, ...]:
    values = _required_list(payload, key)
    parsed: list[int] = []
    for value in values:
        if type(value) is not int:
            message = f"{key} values must be integers"
            raise TypeError(message)
        parsed.append(value)
    return tuple(parsed)


def _overall_status(rows: Sequence[CsvRow]) -> str:
    if rows and all(row["status"] == PASS_STATUS for row in rows):
        return "pretest_pass"
    if any(row["status"] == INELIGIBLE_STATUS for row in rows):
        return "pretest_incomplete"
    if any(row["status"] == PASS_STATUS for row in rows):
        return "pretest_partial"
    return "pretest_skipped"


def _recommendation_sort_key(row: CsvRow) -> tuple[float, float, str]:
    group = 0.0 if row["status"] == PASS_STATUS else 1.0
    return (group, _csv_float_or_inf(row, "steady_step_ms_p50"), row["row_id"])


def _pythonpath_with_current_sys_path(environment: Mapping[str, str]) -> str:
    entries = [str(Path(entry).resolve()) for entry in sys.path if entry]
    inherited = environment.get("PYTHONPATH")
    if inherited:
        entries.extend(
            str(Path(entry).resolve()) for entry in inherited.split(os.pathsep) if entry
        )
    return os.pathsep.join(entries)


def _encode_child_config(config: ChildRowConfig) -> str:
    payload: JsonObject = {
        "config_path": str(config.config_path),
        "output_dir": str(config.output_dir),
        "data_root": config.data_root,
        "row_spec": _row_spec_payload(config.row_spec),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _decode_child_config(encoded: str) -> ChildRowConfig:
    payload = cast(
        "JsonObject",
        json.loads(base64.urlsafe_b64decode(encoded.encode("ascii"))),
    )
    config_path = Path(_required_str(payload, "config_path"))
    resolved = resolve_json_config(config_path)
    settings = _settings(
        resolved,
        data_root_override=_required_str(payload, "data_root"),
    )
    return ChildRowConfig(
        config_path=config_path,
        output_dir=Path(_required_str(payload, "output_dir")),
        data_root=_required_str(payload, "data_root"),
        row_spec=_row_spec_from_payload(_required_object(payload, "row_spec")),
        settings=settings,
    )


def _row_spec_payload(row_spec: RowSpec) -> JsonObject:
    return {
        "row_id": row_spec.row_id,
        "accelerator_mode": row_spec.accelerator_mode,
        "per_device_batch_size": row_spec.per_device_batch_size,
        "precision_policy": row_spec.precision_policy,
        "compile_scope": row_spec.compile_scope,
        "corruption_strategy": row_spec.corruption_strategy,
        "parent_synthetic_row_id": row_spec.parent_synthetic_row_id,
        "candidate_role": row_spec.candidate_role,
        "world_size": row_spec.world_size,
        "nproc_per_node": row_spec.nproc_per_node,
        "cuda_visible_devices": row_spec.cuda_visible_devices,
        "runtime_policy_id": row_spec.runtime_policy_id,
        "memory_format": row_spec.memory_format,
        "autocast_dtype": row_spec.autocast_dtype,
        "fp32_loss": row_spec.fp32_loss,
        "grad_scaler_enabled": row_spec.grad_scaler_enabled,
        "cudnn_benchmark": row_spec.cudnn_benchmark,
        "cudnn_deterministic": row_spec.cudnn_deterministic,
        "deterministic_algorithms": row_spec.deterministic_algorithms,
        "tf32_enabled": row_spec.tf32_enabled,
        "matmul_precision": row_spec.matmul_precision,
        "ddp_static_graph": row_spec.ddp_static_graph,
        "ddp_gradient_as_bucket_view": row_spec.ddp_gradient_as_bucket_view,
        "optimizer_implementation": row_spec.optimizer_implementation,
        "zero_grad_set_to_none": row_spec.zero_grad_set_to_none,
        "gradient_clip_foreach": row_spec.gradient_clip_foreach,
        "compile_dynamic": row_spec.compile_dynamic,
        "optimize_ddp": row_spec.optimize_ddp,
        "compiled_autograd": row_spec.compiled_autograd,
        "reorder_compute_comm_overlap": row_spec.reorder_compute_comm_overlap,
        "ddp_broadcast_buffers": row_spec.ddp_broadcast_buffers,
        "ddp_find_unused_parameters": row_spec.ddp_find_unused_parameters,
        "ddp_bucket_cap_mb": row_spec.ddp_bucket_cap_mb,
        "fused_optimizer": row_spec.fused_optimizer,
    }


def _row_spec_from_payload(payload: JsonObject) -> RowSpec:
    return RowSpec(
        row_id=_required_str(payload, "row_id"),
        accelerator_mode=_required_str(payload, "accelerator_mode"),
        per_device_batch_size=_required_int(payload, "per_device_batch_size"),
        precision_policy=_required_str(payload, "precision_policy"),
        compile_scope=_required_str(payload, "compile_scope"),
        corruption_strategy=_required_str(payload, "corruption_strategy"),
        parent_synthetic_row_id=_required_str(payload, "parent_synthetic_row_id"),
        candidate_role=_required_str(payload, "candidate_role"),
        world_size=_required_int(payload, "world_size"),
        nproc_per_node=_required_int(payload, "nproc_per_node"),
        cuda_visible_devices=_required_str(payload, "cuda_visible_devices"),
        runtime_policy_id=_optional_str(payload, "runtime_policy_id")
        or "fp32_eager_default",
        memory_format=_optional_str(payload, "memory_format") or "contiguous",
        autocast_dtype=_optional_str(payload, "autocast_dtype") or "",
        fp32_loss=_optional_bool(payload, "fp32_loss", default=True),
        grad_scaler_enabled=_optional_bool(
            payload,
            "grad_scaler_enabled",
            default=False,
        ),
        cudnn_benchmark=_optional_bool(payload, "cudnn_benchmark", default=False),
        cudnn_deterministic=_optional_bool(
            payload,
            "cudnn_deterministic",
            default=False,
        ),
        deterministic_algorithms=_optional_bool(
            payload,
            "deterministic_algorithms",
            default=False,
        ),
        tf32_enabled=_optional_bool(payload, "tf32_enabled", default=False),
        matmul_precision=_optional_str(payload, "matmul_precision") or "highest",
        ddp_static_graph=_optional_bool(payload, "ddp_static_graph", default=False),
        ddp_gradient_as_bucket_view=_optional_bool(
            payload,
            "ddp_gradient_as_bucket_view",
            default=False,
        ),
        optimizer_implementation=_optional_str(payload, "optimizer_implementation")
        or "adamw_default",
        zero_grad_set_to_none=_optional_bool(
            payload,
            "zero_grad_set_to_none",
            default=True,
        ),
        gradient_clip_foreach=_optional_bool(
            payload,
            "gradient_clip_foreach",
            default=False,
        ),
        compile_dynamic=_optional_bool(payload, "compile_dynamic", default=False),
        optimize_ddp=_optional_str(payload, "optimize_ddp") or "",
        compiled_autograd=_optional_bool(payload, "compiled_autograd", default=False),
        reorder_compute_comm_overlap=_optional_bool(
            payload,
            "reorder_compute_comm_overlap",
            default=False,
        ),
        ddp_broadcast_buffers=_optional_bool(
            payload,
            "ddp_broadcast_buffers",
            default=True,
        ),
        ddp_find_unused_parameters=_optional_bool(
            payload,
            "ddp_find_unused_parameters",
            default=False,
        ),
        ddp_bucket_cap_mb=_optional_int(payload, "ddp_bucket_cap_mb"),
        fused_optimizer=_optional_bool(payload, "fused_optimizer", default=False),
    )


def _parse_args(argv: Sequence[str] | None) -> ChildProcessArgs:
    parser = argparse.ArgumentParser(description="Real-data runtime pretest helper.")
    parser.add_argument("--child-row")
    parser.add_argument("--ddp-rank-row")
    namespace = parser.parse_args(argv)
    return ChildProcessArgs(
        child_row=_optional_arg_str(namespace, "child_row"),
        ddp_rank_row=_optional_arg_str(namespace, "ddp_rank_row"),
    )


def _optional_arg_str(namespace: argparse.Namespace, key: str) -> str | None:
    value = cast("object", getattr(namespace, key))
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string argument {key!r}"
    raise TypeError(message)


def _cuda_allocated_mb(device: torch.device) -> float:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def _cuda_reserved_mb(device: torch.device) -> float:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)


def _cuda_headroom_fraction(device: torch.device) -> float:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return 1.0
    get_device_properties = cast(
        "CudaDevicePropertiesGetter",
        torch.cuda.get_device_properties,
    )
    properties = get_device_properties(device)
    total = float(properties.total_memory)
    reserved = float(torch.cuda.max_memory_reserved(device))
    if total <= 0.0:
        return 0.0
    return max(0.0, (total - reserved) / total)


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _elapsed_seconds(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000_000.0


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _phase_log_payload(
    *,
    event: str,
    name: str,
    metadata: JsonObject,
    extra: JsonObject | None = None,
) -> JsonObject:
    payload: JsonObject = {
        "event": "eqvae_real_data_pretest_phase",
        "phase_event": event,
        "phase": name,
        "timestamp_utc": _utc_timestamp(),
    }
    if metadata:
        payload["metadata"] = metadata
    if extra:
        payload.update(extra)
    return payload


def _log_phase_event(payload: JsonObject) -> None:
    sys.stderr.write(f"{json.dumps(payload, sort_keys=True)}\n")
    sys.stderr.flush()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(len(sorted_values) * fraction) - 1),
    )
    return sorted_values[index]


def _float_list(payload: JsonObject, key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected list at {key}"
        raise TypeError(message)
    return [_json_float(item, key=key) for item in value]


def _json_float(value: JsonValue, *, key: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric value in {key}"
    raise TypeError(message)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _csv_float_or_inf(row: CsvRow, key: str) -> float:
    value = row[key]
    if not value:
        return math.inf
    try:
        return float(value)
    except ValueError:
        return math.inf


def _csv_int_or_zero(row: CsvRow, key: str) -> int:
    value = row.get(key, "")
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def _csv_has_int(row: CsvRow, key: str) -> bool:
    value = row.get(key, "")
    if not value:
        return False
    try:
        int(value)
    except ValueError:
        return False
    return True


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _rounded_float(value: float) -> float:
    return float(_format_float(value))


def _format_bool(*, value: bool) -> str:
    return "true" if value else "false"


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _recipe_knob_columns(*, row_spec: RowSpec) -> dict[str, str]:
    """Emit the recipe-knob CSV cells from a RowSpec (Spec 0011 S14a).

    Shared by every RUNTIME_MATRIX_COLUMNS producer that owns a RowSpec -- the pretest
    ``_base_row`` and the executor ``_base_selection_row`` -- so an eager RowSpec emits
    the eager-v5 defaults (byte-identical to ``EAGER_RECIPE_KNOB_COLUMNS``) and a
    compiled winner RowSpec emits its measured knobs.

    Returns:
        The seven recipe-knob column cells for ``row_spec``.

    """
    return {
        "optimize_ddp": row_spec.optimize_ddp,
        "compiled_autograd": _format_bool(value=row_spec.compiled_autograd),
        "reorder_compute_comm_overlap": _format_bool(
            value=row_spec.reorder_compute_comm_overlap,
        ),
        "ddp_broadcast_buffers": _format_bool(value=row_spec.ddp_broadcast_buffers),
        "ddp_find_unused_parameters": _format_bool(
            value=row_spec.ddp_find_unused_parameters,
        ),
        "ddp_bucket_cap_mb": _format_optional_int(row_spec.ddp_bucket_cap_mb),
        "fused_optimizer": _format_bool(value=row_spec.fused_optimizer),
    }


def _required_object(payload: JsonObject, key: str) -> JsonObject:
    value = payload.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object at {key}"
    raise TypeError(message)


def _required_list(payload: JsonObject, key: str) -> list[JsonValue]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    message = f"Expected list at {key}"
    raise TypeError(message)


def _required_str(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string at {key}"
    raise TypeError(message)


def _optional_str(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    message = f"Expected optional string at {key}"
    raise TypeError(message)


def _required_str_list(payload: JsonObject, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        message = f"Expected list at {key}"
        raise TypeError(message)
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = f"Expected string list at {key}"
            raise TypeError(message)
        parsed.append(item)
    return parsed


def _required_int(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload.get(key)
    if type(value) is int:
        return value
    message = f"Expected integer at {key}"
    raise TypeError(message)


def _optional_int(payload: JsonObject, key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if type(value) is int:
        return value
    message = f"Expected optional integer at {key}"
    raise TypeError(message)


def _optional_float(payload: JsonObject, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected optional numeric at {key}"
    raise TypeError(message)


def _required_float(payload: JsonObject, key: str) -> float:
    value = payload.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    message = f"Expected numeric at {key}"
    raise TypeError(message)


def _required_bool(payload: JsonObject, key: str) -> bool:
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    message = f"Expected boolean at {key}"
    raise TypeError(message)


def _optional_bool(payload: JsonObject, key: str, *, default: bool) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    message = f"Expected optional boolean at {key}"
    raise TypeError(message)


__all__ = [
    "CORRUPTION_CHECKS_FILENAME",
    "DATALOADER_MATRIX_FILENAME",
    "GATE_HEALTH_FILENAME",
    "GATE_HEALTH_SUMMARY_FILENAME",
    "MANIFEST_FILENAME",
    "NUMERICAL_CHECKS_FILENAME",
    "REAL_DATA_PRETEST_KIND",
    "REAL_DATA_PRETEST_SCHEMA_VERSION",
    "REAL_DATA_PRETEST_SCOPE",
    "REAL_DATA_PRETEST_SOURCE",
    "RECOMMENDATIONS_FILENAME",
    "RUNTIME_MATRIX_FILENAME",
    "RUNTIME_PROOF_FILENAME",
    "RealDataRuntimePretestRequest",
    "write_local_upload_simulation_artifact",
    "write_real_data_runtime_pretest",
]


if __name__ == "__main__":
    raise SystemExit(main())
