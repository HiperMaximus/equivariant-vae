# Copyright 2026 HiperMaximus
"""Benchmark schema and count helpers for spec 0001."""

from __future__ import annotations

from eqvae.benchmarking.dataloader_pretest import (
    LocalDataloaderPretestRequest,
    write_local_dataloader_pretest,
)
from eqvae.benchmarking.model_count import (
    SPEC0001_MODEL_COUNT_TARGET,
    build_model_count_payload,
    write_model_count,
)
from eqvae.benchmarking.model_loss_train_step import (
    LocalModelLossTrainStepRequest,
    write_local_model_loss_train_step,
)
from eqvae.benchmarking.real_data_runtime_pretest import (
    RealDataRuntimePretestRequest,
    write_real_data_runtime_pretest,
)
from eqvae.benchmarking.runtime_schema import (
    EAGER_RECIPE_KNOB_COLUMNS,
    RUNTIME_MATRIX_COLUMNS,
    write_synthetic_benchmark_artifacts,
)
from eqvae.benchmarking.runtime_selection import (
    RuntimeSelectionBenchmarkRequest,
    RuntimeSelectionEvidence,
    write_runtime_selection_benchmark,
)
from eqvae.benchmarking.runtime_selection_executor import (
    RuntimeSelectionExecutionRequest,
    write_runtime_selection_execution,
)
from eqvae.benchmarking.synthetic_timing import (
    REPEAT_SHORTLIST_MEASURED_STEPS,
    REPEAT_SHORTLIST_WARMUP_STEPS,
    SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST,
    SyntheticTimingRequest,
    SyntheticTimingRowSpec,
    build_synthetic_timing_recommendations_payload,
    build_synthetic_timing_runtime_proof_payload,
    compact_synthetic_timing_profile,
    default_synthetic_timing_profile,
    repeat_shortlist_row_specs,
    tiny_upload_simulation_profile,
    write_synthetic_timing_pretest,
)

__all__ = [
    "EAGER_RECIPE_KNOB_COLUMNS",
    "REPEAT_SHORTLIST_MEASURED_STEPS",
    "REPEAT_SHORTLIST_WARMUP_STEPS",
    "RUNTIME_MATRIX_COLUMNS",
    "SPEC0001_MODEL_COUNT_TARGET",
    "SYNTHETIC_TIMING_PHASE_REPEAT_SHORTLIST",
    "LocalDataloaderPretestRequest",
    "LocalModelLossTrainStepRequest",
    "RealDataRuntimePretestRequest",
    "RuntimeSelectionBenchmarkRequest",
    "RuntimeSelectionEvidence",
    "RuntimeSelectionExecutionRequest",
    "SyntheticTimingRequest",
    "SyntheticTimingRowSpec",
    "build_model_count_payload",
    "build_synthetic_timing_recommendations_payload",
    "build_synthetic_timing_runtime_proof_payload",
    "compact_synthetic_timing_profile",
    "default_synthetic_timing_profile",
    "repeat_shortlist_row_specs",
    "tiny_upload_simulation_profile",
    "write_local_dataloader_pretest",
    "write_local_model_loss_train_step",
    "write_model_count",
    "write_real_data_runtime_pretest",
    "write_runtime_selection_benchmark",
    "write_runtime_selection_execution",
    "write_synthetic_benchmark_artifacts",
    "write_synthetic_timing_pretest",
]
