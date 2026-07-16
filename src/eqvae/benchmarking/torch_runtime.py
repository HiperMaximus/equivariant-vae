# Copyright 2026 HiperMaximus
"""Shared torch/CUDA build-version telemetry for Kaggle run artifacts.

The Kaggle GPU environment image is not API-pinnable and drifts with Kaggle's
latest image (see ``docs/decisions/0011-kaggle-code-delivery.md``), so the
reproducibility-correct record of what a run executed on is to MEASURE the torch
build and CUDA toolkit version at run time and stamp them into the run telemetry.
This is the single source both ``benchmarking`` and ``training`` stamp into their
JSON environment blocks.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class TorchRuntimeVersions(TypedDict):
    """The torch build and CUDA toolkit versions a process runs on."""

    torch_version: str
    cuda_version: str | None


def torch_runtime_versions() -> TorchRuntimeVersions:
    """Return the torch build and CUDA toolkit versions this process runs on.

    ``torch.version.cuda`` is ``None`` for a CPU-only build; it is kept as-is so
    the JSON telemetry records ``null`` rather than a fabricated string.

    Returns:
        The torch build version and the CUDA toolkit version.

    """
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
