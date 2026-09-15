# Copyright 2026 HiperMaximus
"""Focused contract and packaging checks for functional-geometry Stage A1."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = REPO_ROOT / "kaggle/kernels/functional_geometry_stage_a1_c4_slq"
CONTRACT_PATH = REPO_ROOT / "docs/data/functional_geometry_stage_a1_contract.json"
HEAVY_STATE_COUNT = 16
METRIC_VECTOR_ROWS = 65_536


def test_stage_a1_contract_locks_the_calibrated_two_gpu_experiment() -> None:
    """Invariant: Stage A1 consumes the result-blind m/r calibration unchanged."""
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["numerics"] == {
        "bootstrap_resamples": 4096,
        "bootstrap_seed": 530078,
        "compile": False,
        "confidence_level": 0.99,
        "jvp_epsilon": 0.008,
        "jvp_fd_direction": "coordinate_rms_one_rademacher",
        "jvp_relative_l2_max": 0.005,
        "lanczos_steps": 64,
        "metric_identity_relative_error_max": 0.002,
        "operator": "Gv=J^T(Jv)",
        "operator_dtype": "float32",
        "probe_batch_size": 4,
        "probe_count": 64,
        "probe_distribution": "normalized_rademacher",
        "probe_seed": 530077,
        "tridiagonal_dtype": "float64",
    }
    assert contract["scope"]["model_device_map"] == {
        "normal_vae": 0,
        "so2_vae": 1,
    }
    assert contract["scope"]["angles_degrees"] == [0, 90, 180, 270]
    assert contract["scope"]["heavy_patch_ranks"] == [0, 12]
    assert contract["scope"]["heavy_state_count"] == HEAVY_STATE_COUNT
    assert contract["resources"]["heavy_metric_vector_rows"] == METRIC_VECTOR_ROWS


def test_stage_a1_kernel_uses_spawned_workers_and_observable_batched_slq() -> None:
    """Invariant: each model has one process/GPU and failures remain diagnosable."""
    source = (KERNEL_ROOT / "run_template.py").read_text(encoding="utf-8")
    metadata = json.loads(
        (KERNEL_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"),
    )

    assert metadata["is_private"] == "true"
    assert metadata["dataset_sources"] == [
        "maximshtefan/eqvae-vae-test-reconstruction-inputs-v1",
        "maximusshtefan/patches-pre-shuffled-ubc-ocean",
    ]
    assert 'mp.get_context("spawn")' in source
    assert "stochastic_lanczos_quadrature_batched" in source
    assert "torch.rot90(patches, turns, (-2, -1))" in source
    assert '"slq_step_complete"' in source
    assert '"worker_failed"' in source
    assert '"traceback": traceback.format_exc()' in source
    assert "compile_operators=False" in source
