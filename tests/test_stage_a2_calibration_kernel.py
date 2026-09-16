# Copyright 2026 HiperMaximus
"""Focused regression tests for the Stage A2 calibration rerun."""

import copy
import math

import pytest
import torch
from experiments import spec0053_stage_a2_calibration as calibration

from eqvae.evaluation.functional_geometry_calibration import (
    affine_chart_latents,
    decoder_path_energy,
    project_line_deviation_,
)


def test_chunked_energy_and_gradient_match_the_monolithic_objective() -> None:
    """Decoder batching changes memory use, not path energy or its gradient."""

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

    def coordinates_to_latents(coordinates: torch.Tensor) -> torch.Tensor:
        return affine_chart_latents(
            left,
            basis,
            coordinates,
            secant_norm=secant_norm,
        )

    def edge_energy(latents: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        decoded = Decoder.decode(latents)
        differences = decoded[1:] - decoded[:-1]
        return segments * differences.flatten(1).square().mean(dim=1).sum()

    chunked = calibration._chunked_path_energy(
        left,
        right,
        endpoint_coordinates=endpoint[0],
        interior_coordinates=initial,
        coordinates_to_latents=coordinates_to_latents,
        edge_energy=edge_energy,
        total_segments=8,
        chunk_segments=4,
        backward=False,
        torch=torch,
    )
    calibration._chunked_path_energy(
        left,
        right,
        endpoint_coordinates=endpoint[0],
        interior_coordinates=chunk_coordinates,
        coordinates_to_latents=coordinates_to_latents,
        edge_energy=edge_energy,
        total_segments=8,
        chunk_segments=4,
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


def test_full_latent_optimizer_uses_direct_normalized_offsets() -> None:
    """The full control needs no materialized identity basis."""

    class Decoder:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return torch.stack((latent[:, 0].square(), latent[:, 1].sin()), dim=1)

    def edge_energy(latents: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        decoded = Decoder.decode(latents)
        differences = decoded[1:] - decoded[:-1]
        return segments * differences.flatten(1).square().mean(dim=1).sum()

    result = calibration._optimizer_probe(
        Decoder(),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[1.0, 0.7]]),
        torch.eye(2),
        coordinate_space="full",
        learning_rate=0.005,
        path_segments=4,
        optimizer_contract={
            "algorithm": "Adam",
            "energy_chunk_segments": 2,
            "iterations": 2,
            "learning_rate_reference_dimension": 2,
            "milestones": [0, 1, 2],
            "trust_deviation_fraction_of_endpoint_secant": 0.2,
        },
        affine_chart_latents=affine_chart_latents,
        eager_edge_energy=edge_energy,
        optimized_edge_energy=edge_energy,
        project_line_deviation_=project_line_deviation_,
        torch=torch,
    )

    assert result["coordinate_space"] == "full"
    assert result["projection_error"] == 0.0
    assert result["status"] == "complete"


def test_full_latent_learning_rate_is_scaled_to_its_dimension() -> None:
    """A full-latent Adam step must not grow with sqrt(number of pixels)."""

    dimension = 16_384

    class Decoder:
        @staticmethod
        def decode(latent: torch.Tensor) -> torch.Tensor:
            return dimension * latent.square()

    def edge_energy(latents: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        decoded = Decoder.decode(latents)
        differences = decoded[1:] - decoded[:-1]
        return segments * differences.flatten(1).square().mean(dim=1).sum()

    result = calibration._optimizer_probe(
        Decoder(),
        torch.zeros(1, dimension),
        torch.ones(1, dimension) / math.sqrt(dimension),
        torch.empty(0),
        coordinate_space="full",
        learning_rate=0.005,
        path_segments=2,
        optimizer_contract={
            "algorithm": "Adam",
            "energy_chunk_segments": 2,
            "iterations": 1,
            "learning_rate_reference_dimension": 32,
            "milestones": [0, 1],
            "trust_deviation_fraction_of_endpoint_secant": 0.2,
        },
        affine_chart_latents=affine_chart_latents,
        eager_edge_energy=edge_energy,
        optimized_edge_energy=edge_energy,
        project_line_deviation_=project_line_deviation_,
        torch=torch,
    )

    assert result["effective_learning_rate"] == pytest.approx(
        0.005 * math.sqrt(32 / dimension)
    )
    assert result["trust_boundary_contact"] is False


def test_reduced_selection_requires_a_valid_paired_full_control() -> None:
    """A cheap reduced candidate cannot pass against a failed full reference."""

    def candidate(space: int | str, energies: tuple[float, float]) -> dict:
        return {
            "best_recorded_energy": min(energies),
            "coordinate_space": space,
            "history": [
                {"energy": energies[0], "iteration": 0},
                {"energy": energies[1], "iteration": 1},
            ],
            "path_segments": 16,
            "status": "complete",
            "trust_boundary_contact": False,
        }

    contract = {
        "numerics": {
            "chart": {"dimensions": [128], "line_parameters": [0.0]},
            "optimizer": {"learning_rate": 0.005, "milestones": [0, 1]},
            "selection": {
                "conditioning_minimum_ratio": 0.001,
                "optimizer_energy_plateau_relative_slack": 0.01,
                "optimizer_minimum_relative_energy_decrease": 0.01,
                "reduced_to_full_best_energy_excess_max": 0.01,
            },
        },
        "scope": {
            "optimizer_workloads": [
                {
                    "candidates": [
                        {"coordinate_spaces": [128, "full"], "path_segments": 16}
                    ],
                    "rank": 4,
                    "route": "prescribed",
                }
            ]
        },
    }
    worker = {
        "model": "normal_vae",
        "workloads": [
            {
                "chart_diagnostics": {
                    "metrics": [
                        {
                            "spectra": [
                                {"dimension": 128, "minimum_to_maximum_ratio": 0.1}
                            ]
                        }
                    ]
                },
                "optimizer_probes": [
                    candidate(128, (1.0, 0.9)),
                    candidate("full", (1.0, 0.9)),
                ],
                "rank": 4,
                "route": "prescribed",
                "status": "complete",
            }
        ],
    }

    valid = calibration._select_numerics([worker], contract)
    assert valid["selected_numerics"]["coordinate_space"] == 128

    invalid_worker = copy.deepcopy(worker)
    invalid_full = invalid_worker["workloads"][0]["optimizer_probes"][1]
    invalid_full["history"][-1]["energy"] = 1.0
    invalid_full["best_recorded_energy"] = 1.0
    invalid = calibration._select_numerics([invalid_worker], contract)
    assert invalid["selected_numerics"]["coordinate_space"] is None
    assert "neither_d128_nor_full_latent_met_the_common_budget" in invalid["blockers"]
