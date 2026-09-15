# Copyright 2026 HiperMaximus
# pyright: reportAny=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Focused regression tests for the Stage A2 calibration rerun."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from eqvae.evaluation.functional_geometry_calibration import (
    affine_chart_latents,
    decoder_path_edge_energy,
    decoder_path_energy,
)

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle/kernels/functional_geometry_stage_a2_calibration"
CONTRACT = ROOT / "docs/data/functional_geometry_stage_a2_calibration_contract.json"


def _template() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stage_a2_template",
        KERNEL / "run_template.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_keeps_the_probe_frozen_and_excludes_final_pilots() -> None:
    """The rerun may tune numerics but cannot train or inspect pilot examples."""
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    scope = contract["scope"]

    assert scope["training_allowed"] is False
    assert scope["scientific_model_comparison"] is False
    assert scope["calibration_patch_ranks"] == [4, 8, 10, 16, 24]
    assert scope["final_pilot_patch_ranks"] == [0, 12]
    assert not set(scope["calibration_patch_ranks"]) & set(
        scope["final_pilot_patch_ranks"],
    )


def test_chunked_energy_and_gradient_match_the_monolithic_objective() -> None:
    """Decoder batching changes memory use, not path energy or its gradient."""
    module = _template()

    class Decoder:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.stack((latent[:, 0].square(), latent[:, 1].sin()), dim=1)

    left = torch.tensor([[0.0, 0.0]])
    right = torch.tensor([[1.0, 0.7]])
    basis = torch.eye(2)
    secant_norm = torch.linalg.vector_norm(right - left)
    times = torch.linspace(0.0, 1.0, 9)[1:-1, None]
    endpoint = (right - left) / secant_norm
    initial = times * endpoint

    chunk_coordinates = initial.clone().requires_grad_()
    chunked = module._chunked_path_energy(  # noqa: SLF001
        Decoder(),
        left,
        right,
        basis=basis,
        interior_coordinates=chunk_coordinates,
        secant_norm=secant_norm,
        total_segments=8,
        chunk_segments=3,
        affine_chart_latents=affine_chart_latents,
        decoder_path_edge_energy=decoder_path_edge_energy,
        backward=True,
        torch=torch,
    )
    assert chunk_coordinates.grad is not None
    chunk_gradient = chunk_coordinates.grad.detach().clone()

    full_coordinates = initial.clone().requires_grad_()
    interior = affine_chart_latents(
        left,
        basis,
        full_coordinates,
        secant_norm=secant_norm,
    )
    decoded = Decoder.decode(torch.cat((left, interior, right), dim=0))
    monolithic = decoder_path_energy(decoded)
    monolithic.backward()

    torch.testing.assert_close(torch.tensor(chunked), monolithic.detach())
    assert full_coordinates.grad is not None
    torch.testing.assert_close(chunk_gradient, full_coordinates.grad)
