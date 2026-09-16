# Copyright 2026 HiperMaximus
# pyright: reportAssignmentType=false
"""Small matrix-free stochastic Lanczos tools for decoder pullback spectra."""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from eqvae.evaluation.functional_geometry_rla import LinearizedDecoder


@dataclass(frozen=True)
class LanczosProbe:
    """Quadrature atoms from one Rademacher probe of a PSD operator."""

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    iterations: int


@dataclass(frozen=True)
class EnergyDimensions:
    """Trace-weighted spectral energy and its approximate required dimensions."""

    total_energy: float
    dimensions_for_90_percent: int
    dimensions_for_95_percent: int
    dimensions_for_99_percent: int


@dataclass(frozen=True)
class StochasticLanczosResult:
    """Deterministic SLQ estimate, retaining quadrature nodes and weights per probe."""

    probes: tuple[LanczosProbe, ...]
    energy: EnergyDimensions


def decoder_metric_matvec(
    operator: LinearizedDecoder,
    directions: Tensor,
    *,
    microbatch: int,
) -> Tensor:
    """Apply ``J.T @ J`` to latent rows without retaining decoder graphs."""
    _validate_rows(directions, name="directions")
    _validate_microbatch(microbatch)
    results: list[Tensor] = []
    for start in range(0, directions.shape[0], microbatch):
        rows = directions[start : start + microbatch].detach()
        jvp = operator.jvp_batch(rows).detach()
        _validate_rows(jvp, name="JVP responses")
        if jvp.shape[0] != rows.shape[0]:
            raise ValueError("JVP responses must preserve the leading batch dimension")
        if jvp.shape[1:] != operator.output.shape[1:]:
            raise ValueError("JVP responses must match decoder output shape")
        vjp = operator.vjp_batch(jvp).detach()
        _validate_rows(vjp, name="VJP responses")
        if vjp.shape != rows.shape:
            raise ValueError("VJP responses must match latent direction shape")
        results.append(vjp)
    return torch.cat(results, dim=0).detach()


def stochastic_lanczos_quadrature(
    matrix_vector: Callable[[Tensor], Tensor],
    *,
    vector_shape: tuple[int, ...],
    probe_count: int,
    lanczos_steps: int,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> StochasticLanczosResult:
    """Estimate a PSD spectrum with deterministic full-reorthogonalized probes."""
    dimension = math.prod(vector_shape)
    if not vector_shape or any(size <= 0 for size in vector_shape):
        raise ValueError("vector shape must contain at least one positive dimension")
    if type(probe_count) is not int or probe_count <= 0:
        raise ValueError("probe count must be a positive plain integer")
    if type(lanczos_steps) is not int or not 0 < lanczos_steps <= dimension:
        raise ValueError("Lanczos steps must be in the closed interval [1, dimension]")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative plain integer")
    if not dtype.is_floating_point:
        raise TypeError("Lanczos dtype must be floating point")

    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)
    probes = tuple(
        _lanczos_probe(
            matrix_vector,
            probe_ordinal=probe_ordinal,
            vector_shape=vector_shape,
            dimension=dimension,
            lanczos_steps=lanczos_steps,
            generator=generator,
            device=resolved_device,
            dtype=dtype,
            progress_callback=progress_callback,
        )
        for probe_ordinal in range(probe_count)
    )
    return StochasticLanczosResult(
        probes=probes,
        energy=_summarize_energy(probes, dimension=dimension),
    )


def stochastic_lanczos_quadrature_batched(
    matrix_vector: Callable[[Tensor], Tensor],
    *,
    vector_shape: tuple[int, ...],
    probe_count: int,
    lanczos_steps: int,
    seed: int,
    probe_batch_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> StochasticLanczosResult:
    """Run independent Lanczos chains together without block-Lanczos coupling."""
    dimension = math.prod(vector_shape)
    if not vector_shape or any(size <= 0 for size in vector_shape):
        raise ValueError("vector shape must contain at least one positive dimension")
    if type(probe_count) is not int or probe_count <= 0:
        raise ValueError("probe count must be a positive plain integer")
    if type(lanczos_steps) is not int or not 0 < lanczos_steps <= dimension:
        raise ValueError("Lanczos steps must be in the closed interval [1, dimension]")
    if type(probe_batch_size) is not int or probe_batch_size <= 0:
        raise ValueError("probe batch size must be a positive plain integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative plain integer")
    if not dtype.is_floating_point:
        raise TypeError("Lanczos dtype must be floating point")

    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    probes: list[LanczosProbe] = []
    for start in range(0, probe_count, probe_batch_size):
        batch_size = min(probe_batch_size, probe_count - start)
        signs = torch.randint(
            0,
            2,
            (batch_size, *vector_shape),
            generator=generator,
            device=resolved_device,
            dtype=torch.int8,
        )
        current = (signs.to(dtype=dtype) * 2 - 1) / math.sqrt(dimension)
        previous = torch.zeros_like(current)
        previous_beta = torch.zeros(batch_size, device=resolved_device, dtype=dtype)
        basis: list[Tensor] = []
        alphas: list[Tensor] = []
        betas: list[Tensor] = []

        for iteration in range(lanczos_steps):
            basis.append(current)
            response = matrix_vector(current).detach()
            _validate_rows(response, name="matrix-vector responses")
            if response.shape != current.shape:
                raise ValueError("matrix-vector responses must match input row shapes")
            if progress_callback is not None:
                progress_callback(start, batch_size, iteration + 1)
            alpha = (current.flatten(1) * response.flatten(1)).sum(dim=1)
            if not bool(torch.isfinite(alpha).all()):
                raise ValueError("Lanczos coefficients must be finite")
            alphas.append(alpha.detach().cpu())
            broadcast = (batch_size, *(1 for _ in vector_shape))
            candidate = response - alpha.reshape(broadcast) * current
            candidate -= previous_beta.reshape(broadcast) * previous
            candidate = _full_reorthogonalize_batched(candidate, basis)
            beta = torch.linalg.vector_norm(candidate.flatten(1), dim=1)
            if not bool(torch.isfinite(beta).all()):
                raise ValueError("Lanczos coefficients must be finite")
            if iteration + 1 == lanczos_steps:
                break
            tolerances = torch.tensor(
                [
                    _breakdown_tolerance(
                        [float(values[row]) for values in alphas],
                        dtype,
                    )
                    for row in range(batch_size)
                ],
                device=resolved_device,
                dtype=dtype,
            )
            if bool((beta <= tolerances).any()):
                raise ValueError(
                    "batched Lanczos chain broke down before requested depth",
                )
            betas.append(beta.detach().cpu())
            previous, current, previous_beta = (
                current,
                candidate / beta.reshape(broadcast),
                beta,
            )

        for row in range(batch_size):
            row_alphas = [float(values[row]) for values in alphas]
            row_betas = [float(values[row]) for values in betas]
            nodes, weights = _quadrature_atoms(
                row_alphas,
                row_betas,
                operator_dtype=dtype,
            )
            probes.append(
                LanczosProbe(
                    nodes=tuple(float(value) for value in nodes.tolist()),
                    weights=tuple(float(value) for value in weights.tolist()),
                    iterations=len(row_alphas),
                ),
            )

    frozen_probes = tuple(probes)
    return StochasticLanczosResult(
        probes=frozen_probes,
        energy=_summarize_energy(frozen_probes, dimension=dimension),
    )


def _lanczos_probe(
    matrix_vector: Callable[[Tensor], Tensor],
    *,
    probe_ordinal: int,
    vector_shape: tuple[int, ...],
    dimension: int,
    lanczos_steps: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    progress_callback: Callable[[int, int], None] | None,
) -> LanczosProbe:
    signs = torch.randint(
        0,
        2,
        vector_shape,
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    current = (signs.to(dtype=dtype) * 2 - 1) / math.sqrt(dimension)
    previous = torch.zeros_like(current)
    previous_beta = 0.0
    basis: list[Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []

    for iteration in range(lanczos_steps):
        basis.append(current)
        response = matrix_vector(current.unsqueeze(0)).detach()
        _validate_rows(response, name="matrix-vector responses")
        if response.shape != (1, *vector_shape):
            raise ValueError("matrix-vector responses must match the input row shape")
        if progress_callback is not None:
            progress_callback(probe_ordinal, iteration + 1)
        candidate = response[0]
        alpha = _scalar_dot(current, candidate)
        if not math.isfinite(alpha):
            raise ValueError("Lanczos coefficients must be finite")
        alphas.append(alpha)
        candidate = candidate - alpha * current - previous_beta * previous
        candidate = _full_reorthogonalize(candidate, basis)
        beta = float(torch.linalg.vector_norm(candidate).detach().cpu())
        if not math.isfinite(beta):
            raise ValueError("Lanczos coefficients must be finite")
        if beta <= _breakdown_tolerance(alphas, dtype):
            break
        if iteration + 1 == lanczos_steps:
            break
        betas.append(beta)
        previous, current, previous_beta = current, candidate / beta, beta

    nodes, weights = _quadrature_atoms(alphas, betas, operator_dtype=dtype)
    return LanczosProbe(
        nodes=tuple(float(value) for value in nodes.tolist()),
        weights=tuple(float(value) for value in weights.tolist()),
        iterations=len(alphas),
    )


def _full_reorthogonalize(candidate: Tensor, basis: list[Tensor]) -> Tensor:
    reorthogonalized = candidate
    for _ in range(2):
        basis_rows = torch.stack(basis).flatten(1)
        coefficients = basis_rows @ reorthogonalized.flatten()
        correction = (coefficients[:, None] * basis_rows).sum(dim=0)
        reorthogonalized -= correction.reshape_as(candidate)
    return reorthogonalized


def _full_reorthogonalize_batched(candidate: Tensor, basis: list[Tensor]) -> Tensor:
    reorthogonalized = candidate
    for _ in range(2):
        basis_rows = torch.stack(basis, dim=1).flatten(2)
        coefficients = torch.bmm(
            basis_rows,
            reorthogonalized.flatten(1).unsqueeze(2),
        ).squeeze(2)
        correction = (coefficients.unsqueeze(2) * basis_rows).sum(dim=1)
        reorthogonalized -= correction.reshape_as(candidate)
    return reorthogonalized


def _quadrature_atoms(
    alphas: list[float],
    betas: list[float],
    *,
    operator_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    diagonal = torch.tensor(alphas, dtype=torch.float64)
    tridiagonal = torch.diag(diagonal)
    if betas:
        off_diagonal = torch.tensor(betas, dtype=torch.float64)
        tridiagonal += torch.diag(off_diagonal, diagonal=1)
        tridiagonal += torch.diag(off_diagonal, diagonal=-1)
    nodes, vectors = torch.linalg.eigh(tridiagonal)
    if not bool(torch.isfinite(nodes).all() and torch.isfinite(vectors).all()):
        raise ValueError("Lanczos quadrature must be finite")
    tolerance = math.sqrt(torch.finfo(operator_dtype).eps) * max(
        1.0,
        float(nodes.abs().max()),
    )
    if bool((nodes < -tolerance).any()):
        raise ValueError("Lanczos operator must be PSD")
    nodes = nodes.clamp_min(0.0)
    weights = vectors[0].square()
    return nodes, weights


def _summarize_energy(
    probes: tuple[LanczosProbe, ...],
    *,
    dimension: int,
) -> EnergyDimensions:
    atoms = [
        (node, dimension * weight / len(probes))
        for probe in probes
        for node, weight in zip(probe.nodes, probe.weights, strict=True)
    ]
    total_energy = sum(node * weight for node, weight in atoms)
    if not math.isfinite(total_energy) or total_energy < 0.0:
        raise ValueError("SLQ energy estimate must be finite and nonnegative")
    if math.isclose(total_energy, 0.0, abs_tol=torch.finfo(torch.float64).tiny):
        return EnergyDimensions(0.0, 0, 0, 0)
    descending = sorted(atoms, key=itemgetter(0), reverse=True)
    return EnergyDimensions(
        total_energy=total_energy,
        dimensions_for_90_percent=_dimension_for_energy(
            descending,
            target_fraction=0.90,
            total_energy=total_energy,
        ),
        dimensions_for_95_percent=_dimension_for_energy(
            descending,
            target_fraction=0.95,
            total_energy=total_energy,
        ),
        dimensions_for_99_percent=_dimension_for_energy(
            descending,
            target_fraction=0.99,
            total_energy=total_energy,
        ),
    )


def _dimension_for_energy(
    atoms: list[tuple[float, float]],
    *,
    target_fraction: float,
    total_energy: float,
) -> int:
    target_energy = target_fraction * total_energy
    accumulated_energy = 0.0
    accumulated_dimension = 0.0
    for node, multiplicity in atoms:
        energy = node * multiplicity
        if accumulated_energy + energy >= target_energy and node > 0.0:
            required = (target_energy - accumulated_energy) / node
            return math.ceil(accumulated_dimension + required)
        accumulated_energy += energy
        accumulated_dimension += multiplicity
    return math.ceil(accumulated_dimension)


def _scalar_dot(left: Tensor, right: Tensor) -> float:
    return float(torch.vdot(left.flatten(), right.flatten()).real.detach().cpu())


def _breakdown_tolerance(alphas: list[float], dtype: torch.dtype) -> float:
    scale = max(1.0, *(abs(value) for value in alphas))
    return math.sqrt(torch.finfo(dtype).eps) * scale


def _validate_microbatch(microbatch: int) -> None:
    if type(microbatch) is not int or microbatch <= 0:
        raise ValueError("microbatch must be a positive plain integer")


def _validate_rows(rows: Tensor, *, name: str) -> None:
    if rows.ndim < 2 or rows.shape[0] <= 0:
        message = f"{name} must contain a nonempty leading row dimension"
        raise ValueError(message)
    if not rows.is_floating_point():
        message = f"{name} must use a floating dtype"
        raise TypeError(message)
    if not bool(torch.isfinite(rows).all()):
        message = f"{name} must be finite"
        raise ValueError(message)
