# Copyright 2026 HiperMaximus
"""CPU invariants for the low-memory decoder metric and SLQ prototype."""

from operator import neg

import pytest
import torch

from eqvae.evaluation.functional_geometry_rla import (
    LinearizedDecoder,
    linearize_decoder,
)
from eqvae.evaluation.functional_geometry_slq import (
    decoder_metric_matvec,
    stochastic_lanczos_quadrature,
    stochastic_lanczos_quadrature_batched,
)

_D90_DIAGONAL = 2
_D95_DIAGONAL = 3
_D99_DIAGONAL = 3
_IDENTITY_D90 = 8
_LOW_RANK_DIMENSION = 2


def test_slq_recovers_a_known_diagonal_spectrum_and_energy_dimensions() -> None:
    """Invariant: full Lanczos recovers diagonal eigenvalues and d90/d95/d99.

    Why: this establishes the quadrature-to-energy conversion before decoder use.
    Constants: derivation from the four supplied diagonal eigenvalues.
    """
    diagonal = torch.tensor((9.0, 3.0, 1.0, 0.1), dtype=torch.float64)

    def matrix_vector(rows: torch.Tensor) -> torch.Tensor:
        return rows * diagonal

    result = stochastic_lanczos_quadrature(
        matrix_vector,
        vector_shape=(4,),
        probe_count=1,
        lanczos_steps=4,
        seed=7,
        dtype=torch.float64,
    )

    assert result.probes[0].nodes == pytest.approx((0.1, 1.0, 3.0, 9.0))
    assert result.probes[0].weights == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert result.energy.total_energy == pytest.approx(13.1)
    assert result.energy.dimensions_for_90_percent == _D90_DIAGONAL
    assert result.energy.dimensions_for_95_percent == _D95_DIAGONAL
    assert result.energy.dimensions_for_99_percent == _D99_DIAGONAL


def test_slq_handles_identity_and_low_rank_psd_operators() -> None:
    """Invariant: PSD degeneracy remains finite and ignores null-energy dimensions.

    Why: decoder pullback metrics can be singular without being numerical failures.
    Constants: derivation from identity size eight and rank-two diagonal values.
    """
    identity = stochastic_lanczos_quadrature(
        lambda rows: rows,
        vector_shape=(8,),
        probe_count=3,
        lanczos_steps=4,
        seed=11,
    )
    low_rank_diagonal = torch.tensor((5.0, 2.0, 0.0, 0.0, 0.0), dtype=torch.float64)
    low_rank = stochastic_lanczos_quadrature(
        lambda rows: rows * low_rank_diagonal,
        vector_shape=(5,),
        probe_count=1,
        lanczos_steps=5,
        seed=11,
        dtype=torch.float64,
    )

    assert identity.energy.total_energy == pytest.approx(8.0)
    assert identity.energy.dimensions_for_90_percent == _IDENTITY_D90
    assert low_rank.energy.total_energy == pytest.approx(7.0)
    assert low_rank.energy.dimensions_for_90_percent == _LOW_RANK_DIMENSION
    assert low_rank.energy.dimensions_for_95_percent == _LOW_RANK_DIMENSION
    assert low_rank.energy.dimensions_for_99_percent == _LOW_RANK_DIMENSION
    assert all(node >= 0.0 for probe in low_rank.probes for node in probe.nodes)
    assert all(weight >= 0.0 for probe in low_rank.probes for weight in probe.weights)


def test_slq_is_seed_deterministic_without_global_rng_state() -> None:
    """Invariant: a local generator gives identical probe atoms for one seed.

    Why: reproducible probes are required for a fixed numerical experiment.
    Constants: seed 31 is a policy-free deterministic test input.
    """
    diagonal = torch.tensor((4.0, 2.0, 1.0), dtype=torch.float64)

    def matrix_vector(rows: torch.Tensor) -> torch.Tensor:
        return rows * diagonal

    progress: list[tuple[int, int]] = []
    first = stochastic_lanczos_quadrature(
        matrix_vector,
        vector_shape=(3,),
        probe_count=3,
        lanczos_steps=3,
        seed=31,
        dtype=torch.float64,
        progress_callback=lambda probe, step: progress.append((probe, step)),
    )
    _ = torch.rand(17)
    second = stochastic_lanczos_quadrature(
        matrix_vector,
        vector_shape=(3,),
        probe_count=3,
        lanczos_steps=3,
        seed=31,
        dtype=torch.float64,
    )

    assert second == first
    assert progress == [(probe, step) for probe in range(3) for step in range(1, 4)]


def test_batched_slq_matches_independent_scalar_chains() -> None:
    """Invariant: batching four probes changes throughput, not their mathematics."""
    diagonal = torch.arange(1, 9, dtype=torch.float64)

    def matrix_vector(rows: torch.Tensor) -> torch.Tensor:
        return rows * diagonal

    progress: list[tuple[int, int, int]] = []
    scalar = stochastic_lanczos_quadrature(
        matrix_vector,
        vector_shape=(8,),
        probe_count=4,
        lanczos_steps=8,
        seed=37,
        dtype=torch.float64,
    )
    batched = stochastic_lanczos_quadrature_batched(
        matrix_vector,
        vector_shape=(8,),
        probe_count=4,
        lanczos_steps=8,
        seed=37,
        probe_batch_size=4,
        dtype=torch.float64,
        progress_callback=lambda start, count, step: progress.append(
            (start, count, step),
        ),
    )

    assert batched.energy.total_energy == pytest.approx(scalar.energy.total_energy)
    assert batched.energy.dimensions_for_90_percent == (
        scalar.energy.dimensions_for_90_percent
    )
    assert batched.energy.dimensions_for_95_percent == (
        scalar.energy.dimensions_for_95_percent
    )
    assert batched.energy.dimensions_for_99_percent == (
        scalar.energy.dimensions_for_99_percent
    )
    for actual, expected in zip(batched.probes, scalar.probes, strict=True):
        assert actual.iterations == expected.iterations
        assert actual.nodes == pytest.approx(expected.nodes)
        assert actual.weights == pytest.approx(expected.weights)
    assert progress == [(0, 4, step) for step in range(1, 9)]


def test_batched_slq_rejects_invalid_probe_batch_size() -> None:
    """Invariant: an empty probe batch cannot silently skip the experiment."""
    with pytest.raises(ValueError, match="batch size"):
        stochastic_lanczos_quadrature_batched(
            lambda rows: rows,
            vector_shape=(2,),
            probe_count=1,
            lanczos_steps=1,
            seed=0,
            probe_batch_size=0,
        )


def test_slq_rejects_nonfinite_and_non_psd_matrix_vector_responses() -> None:
    """Invariant: invalid operators fail instead of producing spectral claims.

    Why: finite PSD quadrature is the minimum condition for metric interpretation.
    Constants: NaN and negative identity are synthetic counterexamples.
    """
    with pytest.raises(ValueError, match="finite"):
        stochastic_lanczos_quadrature(
            lambda rows: torch.full_like(rows, float("nan")),
            vector_shape=(2,),
            probe_count=1,
            lanczos_steps=2,
            seed=0,
        )

    with pytest.raises(ValueError, match="PSD"):
        stochastic_lanczos_quadrature(
            neg,
            vector_shape=(2,),
            probe_count=1,
            lanczos_steps=2,
            seed=0,
        )


def test_decoder_metric_matvec_microbatches_fake_linearized_decoder() -> None:
    """Invariant: JVP then VJP is detached and never exceeds the microbatch.

    Why: this is the memory boundary for applying J.T @ J to latent directions.
    Constants: factors two and three derive an expected factor-six metric map.
    """
    jvp_batches: list[int] = []
    vjp_batches: list[int] = []

    def jvp(rows: torch.Tensor) -> torch.Tensor:
        jvp_batches.append(rows.shape[0])
        assert not rows.requires_grad
        return rows * 2.0

    def vjp(rows: torch.Tensor) -> torch.Tensor:
        vjp_batches.append(rows.shape[0])
        assert not rows.requires_grad
        return rows * 3.0

    operator = LinearizedDecoder(
        output=torch.zeros((1, 3), dtype=torch.float64),
        jvp_batch=jvp,
        vjp_batch=vjp,
    )
    directions = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    directions.requires_grad_()

    result = decoder_metric_matvec(operator, directions, microbatch=2)

    assert torch.equal(result, directions.detach() * 6.0)
    assert not result.requires_grad
    assert jvp_batches == [2, 2, 1]
    assert vjp_batches == [2, 2, 1]


def test_decoder_metric_matvec_matches_real_jtj_and_rejects_nonfinite() -> None:
    """Invariant: the real adapter equals explicit J.T J and rejects NaN output.

    Why: a finite detached pullback is required before measuring a VAE metric.
    Constants: the fixed linear map is a synthetic derivation, not a measurement.
    """
    decoder = torch.nn.Linear(2, 3, bias=False)
    weight = torch.tensor(((1.0, 2.0), (0.0, 3.0), (4.0, 0.0)))
    with torch.no_grad():
        decoder.weight.copy_(weight)
    operator = linearize_decoder(decoder, torch.zeros((1, 2)))
    directions = torch.tensor(((1.0, -1.0), (2.0, 0.5)), requires_grad=True)

    actual = decoder_metric_matvec(operator, directions, microbatch=1)

    assert torch.allclose(actual, directions.detach() @ weight.T @ weight)
    assert not actual.requires_grad

    nonfinite = LinearizedDecoder(
        output=torch.zeros((1, 3)),
        jvp_batch=lambda rows: torch.full(
            (rows.shape[0], 3),
            float("nan"),
        ),
        vjp_batch=lambda rows: rows,
    )
    with pytest.raises(ValueError, match="finite"):
        decoder_metric_matvec(nonfinite, directions, microbatch=1)
