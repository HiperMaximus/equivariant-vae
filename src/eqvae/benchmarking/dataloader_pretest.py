# Copyright 2026 HiperMaximus
"""Measured local CPU dataloader pre-test for spec 0001."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import cache
from multiprocessing import current_process
from multiprocessing.connection import Listener
from typing import TYPE_CHECKING, cast

from torch import Tensor
from torch.utils.data import DataLoader

from eqvae.benchmarking.io import CsvRow, write_csv
from eqvae.benchmarking.runtime_schema import DATALOADER_MATRIX_COLUMNS
from eqvae.config import JsonObject, JsonValue, resolve_json_config
from eqvae.data.dataloaders import PatchTensorDataset, PatchTensorDatasetSpec
from eqvae.data.synthetic import SyntheticPatchSpec, write_synthetic_patch_shard

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

    from eqvae.data.roots import PatchSplit

LOCAL_PRETEST_KIND = "local_synthetic_pretest"
LOCAL_PRETEST_SOURCE = "local_cpu_synthetic_pretest"
LOCAL_CPU = "local_cpu"


class LocalWorkerTransportUnavailableError(RuntimeError):
    """Raised when local multiprocessing workers cannot return tensors."""


@dataclass(frozen=True)
class DataloaderCandidate:
    """One local dataloader pre-test candidate."""

    num_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool
    non_blocking_h2d: bool


@dataclass(frozen=True)
class LocalDataloaderPretestConfig:
    """Resolved local dataloader pre-test configuration."""

    train_count: int
    validation_count: int
    image_size: int
    channels: int
    data_seed: int
    batch_size: int
    warmup_batches: int
    measured_batches: int
    candidates: tuple[DataloaderCandidate, ...]


@dataclass(frozen=True)
class LocalDataloaderPretestRequest:
    """Inputs for the measured local dataloader pre-test."""

    config_path: Path
    output_dir: Path
    run_name: str


@dataclass(frozen=True)
class SyntheticShardWriteRequest:
    """Inputs for writing one synthetic shard for pre-testing."""

    data_dir: Path
    split: PatchSplit
    count: int
    image_size: int
    channels: int
    seed: int
    include_idx: bool


@dataclass(frozen=True)
class CandidateMeasurementRequest:
    """Inputs for measuring one split/candidate pair."""

    benchmark_request: LocalDataloaderPretestRequest
    config: LocalDataloaderPretestConfig
    candidate: DataloaderCandidate
    split: PatchSplit
    bin_path: Path
    csv_path: Path


def write_local_dataloader_pretest(
    request: LocalDataloaderPretestRequest,
) -> Path:
    """Measure the local tensor-only dataloader path and write matrix rows.

    Returns:
        Path to `benchmark/dataloader_matrix.csv`.

    """
    config = _load_pretest_config(request.config_path)
    data_dir = request.output_dir / "data" / "local_synthetic_pretest"
    train_bin, train_csv = _write_split_shard(
        SyntheticShardWriteRequest(
            data_dir=data_dir,
            split="train",
            count=config.train_count,
            image_size=config.image_size,
            channels=config.channels,
            seed=config.data_seed,
            include_idx=False,
        ),
    )
    validation_bin, validation_csv = _write_split_shard(
        SyntheticShardWriteRequest(
            data_dir=data_dir,
            split="validation",
            count=config.validation_count,
            image_size=config.image_size,
            channels=config.channels,
            seed=config.data_seed + 1,
            include_idx=True,
        ),
    )

    rows: list[CsvRow] = []
    split_specs: tuple[tuple[PatchSplit, Path, Path], ...] = (
        ("train", train_bin, train_csv),
        ("validation", validation_bin, validation_csv),
    )
    for split, bin_path, csv_path in split_specs:
        for candidate in config.candidates:
            try:
                row = _measure_candidate(
                    CandidateMeasurementRequest(
                        benchmark_request=request,
                        config=config,
                        candidate=candidate,
                        split=split,
                        bin_path=bin_path,
                        csv_path=csv_path,
                    ),
                )
            except (
                LocalWorkerTransportUnavailableError,
                OSError,
                RuntimeError,
                StopIteration,
                TypeError,
                ValueError,
            ) as exc:
                row = _failure_row(
                    request=request,
                    config=config,
                    candidate=candidate,
                    split=split,
                    failure_kind=_failure_kind(candidate=candidate, error=exc),
                )
            rows.append(row)

    output_path = request.output_dir / "benchmark" / "dataloader_matrix.csv"
    write_csv(output_path, DATALOADER_MATRIX_COLUMNS, rows)
    return output_path


def _load_pretest_config(config_path: Path) -> LocalDataloaderPretestConfig:
    effective = resolve_json_config(config_path).effective_config
    data = _required_object(effective, "data")
    seeds = _required_object(effective, "seeds")
    pretest = _required_object(effective, "dataloader_pretest")
    if _required_str(pretest, "benchmark_kind") != LOCAL_PRETEST_KIND:
        message = "`dataloader_pretest.benchmark_kind` must be local_synthetic_pretest"
        raise ValueError(message)
    if _required_str(pretest, "benchmark_source") != LOCAL_PRETEST_SOURCE:
        message = (
            "`dataloader_pretest.benchmark_source` must be local_cpu_synthetic_pretest"
        )
        raise ValueError(message)
    if _required_bool(pretest, "full_run_eligible"):
        message = "Local dataloader pre-test must not be full-run eligible"
        raise ValueError(message)

    candidates = tuple(
        _parse_candidate(candidate)
        for candidate in _required_sequence(pretest, "candidates")
    )
    if not candidates:
        message = "dataloader_pretest.candidates must not be empty"
        raise ValueError(message)

    config = LocalDataloaderPretestConfig(
        train_count=_required_int(data, "train_samples"),
        validation_count=_required_int(data, "validation_samples"),
        image_size=_required_int(data, "image_size"),
        channels=_required_int(data, "channels"),
        data_seed=_required_int(seeds, "data_seed"),
        batch_size=_required_int(pretest, "batch_size"),
        warmup_batches=_required_int(pretest, "warmup_batches"),
        measured_batches=_required_int(pretest, "measured_batches"),
        candidates=candidates,
    )
    _validate_pretest_config(config)
    return config


def _parse_candidate(raw_candidate: JsonValue) -> DataloaderCandidate:
    if not isinstance(raw_candidate, dict):
        message = "Each dataloader pre-test candidate must be a JSON object"
        raise TypeError(message)
    candidate = cast("JsonObject", raw_candidate)
    parsed = DataloaderCandidate(
        num_workers=_required_int(candidate, "num_workers"),
        prefetch_factor=_optional_int(candidate, "prefetch_factor"),
        pin_memory=_required_bool(candidate, "pin_memory"),
        persistent_workers=_required_bool(candidate, "persistent_workers"),
        non_blocking_h2d=_required_bool(candidate, "non_blocking_h2d"),
    )
    _validate_candidate(parsed)
    return parsed


def _validate_pretest_config(config: LocalDataloaderPretestConfig) -> None:
    _require_positive("train_samples", config.train_count)
    _require_positive("validation_samples", config.validation_count)
    _require_positive("image_size", config.image_size)
    _require_positive("channels", config.channels)
    _require_positive("batch_size", config.batch_size)
    _require_positive("warmup_batches", config.warmup_batches)
    _require_positive("measured_batches", config.measured_batches)
    minimum_samples = config.batch_size * (
        config.warmup_batches + config.measured_batches
    )
    if config.train_count < minimum_samples:
        message = f"train_samples must be at least {minimum_samples}"
        raise ValueError(message)
    if config.validation_count < minimum_samples:
        message = f"validation_samples must be at least {minimum_samples}"
        raise ValueError(message)


def _validate_candidate(candidate: DataloaderCandidate) -> None:
    if candidate.num_workers < 0:
        message = "num_workers must be nonnegative"
        raise ValueError(message)
    if candidate.num_workers == 0 and candidate.prefetch_factor is not None:
        message = "prefetch_factor must be null when num_workers is 0"
        raise ValueError(message)
    if candidate.num_workers > 0 and candidate.prefetch_factor is None:
        message = "prefetch_factor is required when num_workers is positive"
        raise ValueError(message)
    if candidate.prefetch_factor is not None and candidate.prefetch_factor <= 0:
        message = "prefetch_factor must be positive when set"
        raise ValueError(message)
    if candidate.num_workers == 0 and candidate.persistent_workers:
        message = "persistent_workers requires num_workers > 0"
        raise ValueError(message)
    if candidate.pin_memory:
        message = "Local CPU pre-test candidates must keep pin_memory false"
        raise ValueError(message)
    if candidate.non_blocking_h2d:
        message = "Local CPU pre-test candidates must keep non_blocking_h2d false"
        raise ValueError(message)


def _write_split_shard(request: SyntheticShardWriteRequest) -> tuple[Path, Path]:
    bin_path = request.data_dir / f"{request.split}.bin"
    csv_path = request.data_dir / f"{request.split}.csv"
    write_synthetic_patch_shard(
        bin_path=bin_path,
        csv_path=csv_path,
        spec=SyntheticPatchSpec(
            count=request.count,
            image_size=request.image_size,
            channels=request.channels,
            seed=request.seed,
        ),
        include_idx=request.include_idx,
    )
    return bin_path, csv_path


def _measure_candidate(request: CandidateMeasurementRequest) -> CsvRow:
    config = request.config
    candidate = request.candidate
    if candidate.num_workers > 0 and not _worker_transport_available():
        message = "Local multiprocessing tensor transport is unavailable"
        raise LocalWorkerTransportUnavailableError(message)
    dataset = PatchTensorDataset(
        PatchTensorDatasetSpec(
            bin_path=request.bin_path,
            csv_path=request.csv_path,
            split=request.split,
            image_size=config.image_size,
            channels=config.channels,
            validate_crc=False,
        ),
    )
    loader = _make_loader(dataset=dataset, config=config, candidate=candidate)
    iterator: Iterator[object] = iter(loader)
    measured_fetch_ms: list[float] = []
    measured_samples = 0
    try:
        for _ in range(config.warmup_batches):
            _next_tensor(iterator)
        for _ in range(config.measured_batches):
            start_ns = time.perf_counter_ns()
            batch = _next_tensor(iterator)
            elapsed_ns = time.perf_counter_ns() - start_ns
            measured_fetch_ms.append(elapsed_ns / 1_000_000.0)
            measured_samples += int(batch.shape[0])
    finally:
        del iterator
        del loader
        dataset.close()

    total_fetch_sec = sum(measured_fetch_ms) / 1000.0
    loader_samples_sec = (
        0.0 if total_fetch_sec <= 0.0 else measured_samples / total_fetch_sec
    )
    return {
        "run_name": request.benchmark_request.run_name,
        "benchmark_kind": LOCAL_PRETEST_KIND,
        "benchmark_source": LOCAL_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": LOCAL_CPU,
        "machine_shape": LOCAL_CPU,
        "world_size": "1",
        "rank": "0",
        "split": request.split,
        "num_workers": str(candidate.num_workers),
        "prefetch_factor": _format_optional_int(candidate.prefetch_factor),
        "pin_memory": _format_bool(value=candidate.pin_memory),
        "persistent_workers": _format_bool(value=candidate.persistent_workers),
        "non_blocking_h2d": _format_bool(value=candidate.non_blocking_h2d),
        "batch_size": str(config.batch_size),
        "batches_measured": str(len(measured_fetch_ms)),
        "batch_fetch_ms_p50": _format_float(_percentile(measured_fetch_ms, 0.50)),
        "batch_fetch_ms_p95": _format_float(_percentile(measured_fetch_ms, 0.95)),
        "h2d_ms_p50": "",
        "h2d_ms_p95": "",
        "loader_samples_sec": _format_float(loader_samples_sec),
        "trainer_samples_sec": "",
        "data_wait_fraction_p50": "",
        "data_wait_fraction_p95": "",
        "rank_sample_count": str(measured_samples),
        "dropped_sample_count": "0",
        "status": "local_pass",
        "failure_kind": "",
    }


def _failure_row(
    *,
    request: LocalDataloaderPretestRequest,
    config: LocalDataloaderPretestConfig,
    candidate: DataloaderCandidate,
    split: PatchSplit,
    failure_kind: str,
) -> CsvRow:
    return {
        "run_name": request.run_name,
        "benchmark_kind": LOCAL_PRETEST_KIND,
        "benchmark_source": LOCAL_PRETEST_SOURCE,
        "full_run_eligible": "false",
        "accelerator_mode": LOCAL_CPU,
        "machine_shape": LOCAL_CPU,
        "world_size": "1",
        "rank": "0",
        "split": split,
        "num_workers": str(candidate.num_workers),
        "prefetch_factor": _format_optional_int(candidate.prefetch_factor),
        "pin_memory": _format_bool(value=candidate.pin_memory),
        "persistent_workers": _format_bool(value=candidate.persistent_workers),
        "non_blocking_h2d": _format_bool(value=candidate.non_blocking_h2d),
        "batch_size": str(config.batch_size),
        "batches_measured": "0",
        "batch_fetch_ms_p50": "",
        "batch_fetch_ms_p95": "",
        "h2d_ms_p50": "",
        "h2d_ms_p95": "",
        "loader_samples_sec": "",
        "trainer_samples_sec": "",
        "data_wait_fraction_p50": "",
        "data_wait_fraction_p95": "",
        "rank_sample_count": "0",
        "dropped_sample_count": "0",
        "status": "fail",
        "failure_kind": failure_kind,
    }


def _failure_kind(*, candidate: DataloaderCandidate, error: Exception) -> str:
    if isinstance(error, LocalWorkerTransportUnavailableError):
        return "local_worker_transport_unavailable"
    if candidate.num_workers > 0:
        return "local_worker_error"
    return f"local_loader_error:{type(error).__name__}"


def _make_loader(
    *,
    dataset: PatchTensorDataset,
    config: LocalDataloaderPretestConfig,
    candidate: DataloaderCandidate,
) -> DataLoader[Tensor]:
    if candidate.num_workers == 0:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
    prefetch_factor = candidate.prefetch_factor
    if prefetch_factor is None:
        message = "prefetch_factor is required when num_workers is positive"
        raise ValueError(message)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=candidate.num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=candidate.pin_memory,
        persistent_workers=candidate.persistent_workers,
        timeout=5.0,
    )


def _next_tensor(iterator: Iterator[object]) -> Tensor:
    batch = next(iterator)
    if isinstance(batch, Tensor):
        return batch
    message = f"Expected dataloader batch tensor, got {type(batch).__name__}"
    raise TypeError(message)


@cache
def _worker_transport_available() -> bool:
    try:
        listener = Listener(authkey=current_process().authkey, backlog=1)
    except OSError:
        return False
    listener.close()
    return True


def _percentile(values: Iterable[float], quantile: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        message = "Cannot compute percentile of an empty sequence"
        raise ValueError(message)
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def _required_object(mapping: Mapping[str, JsonValue], key: str) -> JsonObject:
    value = mapping.get(key)
    if isinstance(value, dict):
        return cast("JsonObject", value)
    message = f"Expected object config field `{key}`"
    raise TypeError(message)


def _required_sequence(
    mapping: Mapping[str, JsonValue],
    key: str,
) -> tuple[JsonValue, ...]:
    value = mapping.get(key)
    if isinstance(value, list):
        return tuple(value)
    message = f"Expected list config field `{key}`"
    raise TypeError(message)


def _required_str(mapping: Mapping[str, JsonValue], key: str) -> str:
    value = mapping.get(key)
    if isinstance(value, str):
        return value
    message = f"Expected string config field `{key}`"
    raise TypeError(message)


def _required_int(mapping: Mapping[str, JsonValue], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected integer config field `{key}`"
    raise TypeError(message)


def _optional_int(mapping: Mapping[str, JsonValue], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    message = f"Expected nullable integer config field `{key}`"
    raise TypeError(message)


def _required_bool(mapping: Mapping[str, JsonValue], key: str) -> bool:
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    message = f"Expected boolean config field `{key}`"
    raise TypeError(message)


def _require_positive(name: str, value: int) -> None:
    if value > 0:
        return
    message = f"{name} must be positive"
    raise ValueError(message)


def _format_bool(*, value: bool) -> str:
    return "true" if value else "false"


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


__all__ = [
    "LocalDataloaderPretestRequest",
    "write_local_dataloader_pretest",
]
