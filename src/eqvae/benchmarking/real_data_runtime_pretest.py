# Copyright 2026 HiperMaximus
"""Non-promotable capped real-data runtime pretest artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess  # noqa: S404
import sys
import time
import zlib
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from eqvae.benchmarking.io import CsvRow, JsonObject, JsonValue, write_csv, write_json
from eqvae.benchmarking.runtime_schema import (
    CORRUPTION_CHECK_COLUMNS,
    DATALOADER_MATRIX_COLUMNS,
    GATE_HEALTH_COLUMNS,
    NUMERICAL_CHECK_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
)
from eqvae.config import ResolvedConfig, resolve_json_config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import torch

    from eqvae.corruption.stain import StainCorruptionProfile, StainCorruptionResult
    from eqvae.data.patch_shards import PatchRecord
    from eqvae.data.roots import PatchDataPaths, PatchSplitPaths
    from eqvae.data.training_batches import PatchTrainingBatch
    from eqvae.models.non_equivariant_vae import NonEquivariantVAE
    from eqvae.training.step import TrainStepRequest, TrainStepResult

REAL_DATA_PRETEST_SCHEMA_VERSION = "spec0001.real_data_runtime_pretest.v1"
REAL_DATA_PRETEST_KIND = "real_data_runtime_pretest"
REAL_DATA_PRETEST_SOURCE = "kaggle_capped_real_data_train_step_pretest"
REAL_DATA_PRETEST_SCOPE = "non_promotable_real_data_runtime_pretest"
EXPECTED_DATASET_SLUG = "maximusshtefan/patches-pre-shuffled-ubc-ocean"
EXPECTED_REAL_TRAIN_PATCH_COUNT = 300_000
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
SELECTED_RUNTIME_FILENAME = "selected_runtime.json"
SINGLE_VISIBLE_T4 = "single_visible_t4"
DUAL_T4_DDP = "dual_t4_ddp"
AMP_OFF_FP32 = "amp_off_fp32"
COMPILE_NONE = "none"
BRANCHLESS_ALL = "branchless_all"
INDEXED_MASKED = "indexed_masked"
PASS_STATUS = "pass"  # noqa: S105
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
ADAM_BETA_COUNT = 2
LATENT_DOWNSAMPLE_FACTOR = 8
MODEL_INPUT_SHAPE_NDIM = 4
FILE_HASH_CHUNK_BYTES = 8 * 1024 * 1024
VALIDATION_CLEAN_PROOF_BATCH_SIZE = 12


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


def write_real_data_runtime_pretest(request: RealDataRuntimePretestRequest) -> Path:
    """Run the capped pretest surface and write non-promotable artifacts.

    Returns:
        Path to `benchmark/real_data_runtime_pretest_recommendations.json`.

    """
    resolved = resolve_json_config(request.config_path)
    settings = _settings(resolved, data_root_override=request.data_root)
    benchmark_dir = request.output_dir / "benchmark"
    metrics_dir = request.output_dir / "metrics"
    request.output_dir.mkdir(parents=True, exist_ok=True)
    _reject_selected_runtime_artifact(request.output_dir)
    data_proof = _real_data_identity_and_clean_path_proof(settings)
    rows = _run_stage1_rows(
        request=request,
        settings=settings,
        row_specs=_stage1_row_specs(settings),
    )
    dataloader_rows = _schema_dataloader_rows(settings=settings, data_proof=data_proof)
    numerical_rows = _schema_numerical_rows(settings=settings, rows=rows)
    corruption_rows = _schema_corruption_rows(settings=settings, rows=rows)

    write_json(
        benchmark_dir / MANIFEST_FILENAME,
        _manifest_payload(
            request=request,
            resolved=resolved,
            settings=settings,
            data_proof=data_proof,
        ),
    )
    write_json(
        benchmark_dir / RUNTIME_PROOF_FILENAME,
        _runtime_proof_payload(settings=settings, rows=rows, data_proof=data_proof),
    )
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
    write_csv(metrics_dir / GATE_HEALTH_FILENAME, GATE_HEALTH_COLUMNS, ())
    write_json(
        benchmark_dir / GATE_HEALTH_SUMMARY_FILENAME,
        _gate_health_summary_payload(),
    )
    recommendations_path = benchmark_dir / RECOMMENDATIONS_FILENAME
    write_json(
        recommendations_path,
        _recommendations_payload(settings=settings, rows=rows),
    )
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
        message = "Expected --child-row for helper mode"
        raise ValueError(message)
    _run_child_row(_decode_child_config(args.child_row))
    return 0


def _run_stage1_rows(
    *,
    request: RealDataRuntimePretestRequest,
    settings: RealDataRuntimePretestSettings,
    row_specs: Sequence[RowSpec],
) -> list[CsvRow]:
    rows: list[CsvRow] = []
    for row_spec in row_specs:
        if row_spec.precision_policy != AMP_OFF_FP32:
            rows.append(
                _unsupported_row(
                    settings=settings,
                    row_spec=row_spec,
                    failure_kind="amp_followup_requires_stable_fp32_candidates",
                ),
            )
            continue
        if row_spec.compile_scope != COMPILE_NONE:
            rows.append(
                _unsupported_row(
                    settings=settings,
                    row_spec=row_spec,
                    failure_kind="compile_scope_measurement_pending",
                ),
            )
            continue
        if row_spec.accelerator_mode != SINGLE_VISIBLE_T4:
            rows.append(
                _unsupported_row(
                    settings=settings,
                    row_spec=row_spec,
                    failure_kind="dual_t4_ddp_measurement_pending",
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


def _run_child_row(config: ChildRowConfig) -> None:  # noqa: PLR0914, PLR0915
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
    from eqvae.models.non_equivariant_vae import (  # noqa: PLC0415
        LATENT_CHANNELS,
        build_non_equivariant_vae,
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
    model = build_non_equivariant_vae(
        norm_groups=config.settings.norm_groups,
    ).to(device)
    optimizer, _summary = create_adamw_optimizer(
        model,
        config=SpecAdamWConfig(
            learning_rate=config.settings.learning_rate,
            weight_decay=config.settings.weight_decay,
            gate_lr_multiplier=1.0,
            gradient_clip_global_norm=config.settings.gradient_clip_global_norm,
            beta1=0.9,
            beta2=0.999,
        ),
    )
    profile = profile_from_config(config.settings.corruption_config)
    iterator = iter(loader)
    step_ms: list[float] = []
    samples = 0
    try:  # noqa: PLW0717
        for step_index in range(config.settings.warmup_steps):
            _run_one_train_batch(
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
                latent_channels=LATENT_CHANNELS,
                beta_for_step_fn=beta_for_step,
                train_step_request_factory=TrainStepRequest,
                run_train_step_fn=run_train_step,
            )
        torch.cuda.reset_peak_memory_stats(device)
        for step_index in range(config.settings.measured_steps):
            start_ns = time.perf_counter_ns()
            batch_size = _run_one_train_batch(
                iterator=iterator,
                model=model,
                optimizer=optimizer,
                device=device,
                profile=profile,
                normalize_uint8_batch_fn=normalize_uint8_batch,
                corrupt_normalized_batch_fn=corrupt_normalized_batch,
                settings=config.settings,
                step_index=step_index + config.settings.warmup_steps,
                row_spec=row_spec,
                latent_channels=LATENT_CHANNELS,
                beta_for_step_fn=beta_for_step,
                train_step_request_factory=TrainStepRequest,
                run_train_step_fn=run_train_step,
            )
            torch.cuda.synchronize(device)
            step_ms.append(_elapsed_ms(start_ns))
            samples += batch_size
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
    except FileNotFoundError as exc:
        return _data_proof_failure_payload(
            settings=settings,
            status=SKIPPED_UNSUPPORTED,
            failure_kind="data_root_unavailable",
            failure_message=str(exc),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return _data_proof_failure_payload(
            settings=settings,
            status="fail",
            failure_kind=f"data_proof_{type(exc).__name__}",
            failure_message=str(exc),
        )


def _real_data_identity_and_clean_path_proof_or_raise(
    settings: RealDataRuntimePretestSettings,
) -> JsonObject:
    from eqvae.data.roots import resolve_patch_data_paths  # noqa: PLC0415

    paths = resolve_patch_data_paths(settings.data_root)
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
        "file_hashes": [],
        "splits": {},
        "clean_validation_dataloader": {
            "status": status,
            "failure_kind": failure_kind,
        },
        "failure_kind": failure_kind,
        "failure_message_hash": _hash_text(failure_message),
    }


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


def _manifest_payload(
    *,
    request: RealDataRuntimePretestRequest,
    resolved: ResolvedConfig,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
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
            "real_data_proof": data_proof,
            "timed_rows_eligible": False,
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
                GATE_HEALTH_SUMMARY_FILENAME,
                RECOMMENDATIONS_FILENAME,
            ],
            "selected_runtime_written": False,
        },
    )


def _runtime_proof_payload(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
    data_proof: JsonObject,
) -> JsonObject:
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
                    if row["status"] == INELIGIBLE_STATUS
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
            "compile_settle_policy": {
                "compile_settle_steps": settings.compile_settle_steps,
                "counter_source": "torch._dynamo.utils.counters_with_reset_per_row",
                "implemented_in_this_runner": False,
            },
            "selection_ready": False,
            "selected_runtime_written": False,
            "evidence_gate": _runtime_evidence_gate(data_proof),
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


def _runtime_evidence_gate(data_proof: JsonObject) -> str:
    if (
        _required_str(data_proof, "identity_status") == PASS_STATUS
        and _required_str(data_proof, "crc_validation_status") == PASS_STATUS
        and _required_str(data_proof, "window_status") == PASS_STATUS
        and _required_str(data_proof, "clean_validation_dataloader_status")
        == PASS_STATUS
    ):
        return (
            "Real-data identity, CRC, train/validation-window, and clean "
            "validation dataloader evidence passed. Timing rows remain "
            "ineligible until real dataloader-throughput, numerical, "
            "corruption, gate-health, DDP, and compile-settle evidence passes."
        )
    return (
        "Timing rows remain ineligible until real-data identity, CRC, "
        "validation-window, dataloader, numerical, corruption, gate-health, "
        "DDP, and compile-settle evidence passes."
    )


def _schema_dataloader_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    data_proof: JsonObject,
) -> list[CsvRow]:
    persistent_workers = _format_bool(value=DEFAULT_DATALOADER_PERSISTENT_WORKERS)
    non_blocking_h2d = _format_bool(value=DEFAULT_DATALOADER_NON_BLOCKING_H2D)
    clean_validation = _required_object(data_proof, "clean_validation_dataloader")
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
) -> list[CsvRow]:
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
            "status": SKIPPED_UNSUPPORTED,
            "failure_kind": "paired_numerical_checks_pending",
        }
        for row in rows
    ]


def _schema_corruption_rows(
    *,
    settings: RealDataRuntimePretestSettings,
    rows: Sequence[CsvRow],
) -> list[CsvRow]:
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
            "status": SKIPPED_UNSUPPORTED,
            "failure_kind": "corruption_equivalence_checks_pending",
        }
        for row in rows
    ]


def _gate_health_summary_payload() -> JsonObject:
    return {
        "status": SKIPPED_UNSUPPORTED,
        "benchmark_kind": REAL_DATA_PRETEST_KIND,
        "benchmark_source": REAL_DATA_PRETEST_SOURCE,
        "overall_status": SKIPPED_UNSUPPORTED,
        "full_run_eligible": False,
        "logged_intervals": 0,
        "module_count": 0,
        "nonfinite_count": None,
        "failing_modules": [],
        "warning_modules": [],
        "notes": "Gate-health measurement pending for capped real-data pretest.",
    }


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
    )


def _parse_args(argv: Sequence[str] | None) -> ChildProcessArgs:
    parser = argparse.ArgumentParser(description="Real-data runtime pretest helper.")
    parser.add_argument("--child-row")
    namespace = parser.parse_args(argv)
    return ChildProcessArgs(child_row=_optional_arg_str(namespace, "child_row"))


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


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _format_bool(*, value: bool) -> str:
    return "true" if value else "false"


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
