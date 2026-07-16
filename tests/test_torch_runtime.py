# Copyright 2026 HiperMaximus
"""Tests for the shared torch/CUDA runtime-version telemetry helper."""

from __future__ import annotations

import torch

from eqvae.benchmarking.torch_runtime import torch_runtime_versions


def test_torch_runtime_versions_stamps_the_running_torch_and_cuda_build() -> None:
    """The helper reports the running torch build and CUDA toolkit version."""
    versions = torch_runtime_versions()
    assert set(versions) == {"torch_version", "cuda_version"}
    assert versions["torch_version"] == str(torch.__version__)
    assert versions["cuda_version"] == torch.version.cuda
