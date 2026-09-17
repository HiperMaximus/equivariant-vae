# Copyright 2026 HiperMaximus
# PyTorch's jvp stubs include the optional has_aux return in their union.
# pyright: reportAssignmentType=false
"""Small fixed-chart numerical pieces for the Stage A2 geometry pilot."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

_PATH_SEGMENTS = 32
_NUMERIC_RANK_RTOL = 32 * torch.finfo(torch.float32).eps
_JVP_MICROBATCH = 4


def construct_path_chart(
    decoder: Callable[[Tensor], Tensor],
    path: Tensor,
) -> dict[str, object]:
    """Construct the frozen decoder-visible chart from one full-latent path."""
    secants = (path[1:] - path[:-1]).flatten(1).to(torch.float64)[:_PATH_SEGMENTS]
    norms = torch.linalg.vector_norm(secants, dim=1)
    secants = secants[norms > 0.0]
    latent_shape = tuple(path.shape[1:])
    empty_basis = path.new_empty((0, *latent_shape))
    empty_spectrum = torch.empty(0, device=path.device, dtype=torch.float64)
    empty_knot_spectra = torch.empty(
        (path.shape[0], 0), device=path.device, dtype=torch.float64
    )

    if secants.shape[0] == 0:
        return {
            "U0": empty_basis,
            "inactive": empty_basis,
            "path_basis": empty_basis,
            "path_singular_values": empty_spectrum,
            "aggregate_eigenvalues": empty_spectrum,
            "aggregate_singular_values": empty_spectrum,
            "knot_singular_values": empty_knot_spectra,
            "knot_metric_variation": empty_spectrum,
            "knot_rank_valid": torch.zeros(
                path.shape[0], device=path.device, dtype=torch.bool
            ),
            "path_rank": 0,
            "rank": 0,
            "condition": torch.tensor(
                float("inf"),
                device=path.device,
                dtype=torch.float64,
            ),
            "numeric_floor": torch.zeros((), device=path.device, dtype=torch.float64),
        }

    normalized = secants / norms[norms > 0.0, None]
    _, path_singular_values, right_vectors = torch.linalg.svd(
        normalized,
        full_matrices=False,
    )
    path_floor = _numeric_floor(path_singular_values)
    path_rank = int((path_singular_values > path_floor).sum())
    basis_flat = right_vectors[:path_rank]
    basis = basis_flat.reshape(path_rank, *latent_shape).to(dtype=path.dtype)

    if path_rank == 0:
        return {
            "U0": empty_basis,
            "inactive": empty_basis,
            "path_basis": empty_basis,
            "path_singular_values": path_singular_values,
            "aggregate_eigenvalues": empty_spectrum,
            "aggregate_singular_values": empty_spectrum,
            "knot_singular_values": empty_knot_spectra,
            "knot_metric_variation": empty_spectrum,
            "knot_rank_valid": torch.zeros(
                path.shape[0], device=path.device, dtype=torch.bool
            ),
            "path_rank": 0,
            "rank": 0,
            "condition": torch.tensor(
                float("inf"),
                device=path.device,
                dtype=torch.float64,
            ),
            "numeric_floor": torch.zeros((), device=path.device, dtype=torch.float64),
        }

    gram = torch.zeros((path_rank, path_rank), device=path.device, dtype=torch.float64)
    knot_grams = []
    output_numel = 0
    for knot in path:
        responses = _decoder_jvp(decoder, knot, basis)
        output_numel = responses[0].numel()
        rows = responses.flatten(1).to(torch.float64)
        knot_gram = rows @ rows.T / output_numel
        knot_grams.append(knot_gram)
        gram += knot_gram
    gram /= path.shape[0]
    eigenvalues, visibility_vectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.flip(0)
    visibility_vectors = visibility_vectors.flip(1)
    aggregate_singular_values = eigenvalues.clamp_min(0.0).sqrt()
    numeric_floor = _numeric_floor(aggregate_singular_values)
    retained = aggregate_singular_values > numeric_floor

    retained_vectors = visibility_vectors[:, retained]
    visible_flat = retained_vectors.T @ basis_flat
    inactive_flat = visibility_vectors[:, ~retained].T @ basis_flat
    visible = _orient_rows(visible_flat).reshape(
        visible_flat.shape[0],
        *latent_shape,
    ).to(path.dtype)
    inactive = _orient_rows(inactive_flat).reshape(
        inactive_flat.shape[0],
        *latent_shape,
    ).to(path.dtype)
    rank = int(retained.sum())
    if rank:
        chart_grams = torch.stack(
            [retained_vectors.T @ knot_gram @ retained_vectors for knot_gram in knot_grams]
        )
        knot_singular_values = torch.linalg.eigvalsh(chart_grams).flip(1).clamp_min(0).sqrt()
        knot_metric_variation = torch.linalg.matrix_norm(
            chart_grams[1:] - chart_grams[:-1]
        ) / torch.linalg.matrix_norm(chart_grams[:-1]).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        knot_rank_valid = (knot_singular_values > numeric_floor).all(dim=1)
    else:
        knot_singular_values = empty_knot_spectra
        knot_metric_variation = empty_spectrum
        knot_rank_valid = torch.zeros(
            path.shape[0], device=path.device, dtype=torch.bool
        )
    condition = (
        aggregate_singular_values[0] / aggregate_singular_values[rank - 1]
        if rank
        else torch.tensor(float("inf"), device=path.device, dtype=torch.float64)
    )
    return {
        "U0": visible,
        "inactive": inactive,
        "path_basis": basis,
        "path_singular_values": path_singular_values,
        "aggregate_eigenvalues": eigenvalues,
        "aggregate_singular_values": aggregate_singular_values,
        "knot_singular_values": knot_singular_values,
        "knot_metric_variation": knot_metric_variation,
        "knot_rank_valid": knot_rank_valid,
        "path_rank": path_rank,
        "rank": rank,
        "condition": condition,
        "numeric_floor": numeric_floor,
    }


def estimate_initial_velocity(path: Tensor, U0: Tensor) -> Tensor:
    """Use the frozen second-order forward difference for the first-side Log candidate."""
    basis = U0.flatten(1)
    first = (path[1] - path[0]).flatten()
    second = (path[2] - path[0]).flatten()
    return (4.0 * (basis @ first) - basis @ second) / (2.0 / _PATH_SEGMENTS)


def shooting_consistency_rollout(
    decoder: Callable[[Tensor], Tensor],
    z0: Tensor,
    U0: Tensor,
    c0: Tensor,
    aggregate_sigma_max: Tensor,
    *,
    target: Tensor | None = None,
) -> dict[str, object]:
    """Run the fixed 8/16-step free IVPs without any endpoint refitting."""
    return {
        "primary": _integrate_shooting(
            decoder,
            z0,
            U0,
            c0,
            aggregate_sigma_max,
            steps_per_quarter=8,
            target=target,
        ),
        "refined": _integrate_shooting(
            decoder,
            z0,
            U0,
            c0,
            aggregate_sigma_max,
            steps_per_quarter=16,
            target=target,
        ),
    }


def transport_rollout(
    decoder: Callable[[Tensor], Tensor],
    latents: Tensor,
    U0: Tensor,
    aggregate_sigma_max: Tensor,
    initial_tangent: Tensor,
) -> dict[str, object]:
    """Transport one tangent and the complete chart frame along a fixed rollout."""
    rank = U0.shape[0]
    if rank == 0:
        return _undefined_transport(U0, initial_tangent)

    current = _thin_decoder_jacobian(decoder, latents[0], U0)
    current_svd = _thin_svd(current, aggregate_sigma_max)
    if current_svd is None:
        return _undefined_transport(U0, initial_tangent)
    tangent = initial_tangent.detach().clone()
    frame = _inverse_square_root(current_svd).to(dtype=U0.dtype)
    tangents = [tangent]
    frames = [frame]
    maps: list[Tensor] = []
    singular_values = [current_svd[1]]
    for latent in latents[1:]:
        following = _thin_decoder_jacobian(decoder, latent, U0)
        following_svd = _thin_svd(following, aggregate_sigma_max)
        if following_svd is None:
            return {
                "intrinsic_status": "undefined",
                "coordinates": torch.stack(tangents),
                "frames": torch.stack(frames),
                "maps": torch.stack(maps) if maps else U0.new_empty((0, rank, rank)),
                "singular_values": torch.stack(singular_values),
                "return_tangent": tangents[-1],
                "return_frame": frames[-1],
            }
        transition = _pseudoinverse(following_svd) @ _normalized_columns(current)
        tangent = (transition @ tangent.to(torch.float64)).to(dtype=U0.dtype)
        frame = (transition @ frame.to(torch.float64)).to(dtype=U0.dtype)
        maps.append(transition.to(dtype=U0.dtype))
        singular_values.append(following_svd[1])
        tangents.append(tangent)
        frames.append(frame)
        current = following
    return {
        "intrinsic_status": "defined",
        "coordinates": torch.stack(tangents),
        "frames": torch.stack(frames),
        "maps": torch.stack(maps) if maps else U0.new_empty((0, rank, rank)),
        "singular_values": torch.stack(singular_values),
        "return_tangent": tangents[-1],
        "return_frame": frames[-1],
    }


def _integrate_shooting(
    decoder: Callable[[Tensor], Tensor],
    z0: Tensor,
    U0: Tensor,
    c0: Tensor,
    aggregate_sigma_max: Tensor,
    *,
    steps_per_quarter: int,
    target: Tensor | None,
) -> dict[str, object]:
    coordinates = torch.zeros_like(c0)
    velocity = c0.detach().clone()
    points = [z0.detach().clone()]
    coordinate_history = [coordinates]
    velocity_history = [velocity]
    step_size = 1.0 / steps_per_quarter
    status = "defined"

    if U0.shape[0] == 0:
        status = "undefined"

    for _ in range(4 * steps_per_quarter) if status == "defined" else ():
        acceleration = _chart_acceleration(
            decoder,
            _chart_point(z0, U0, coordinates),
            U0,
            velocity,
            aggregate_sigma_max,
        )
        if acceleration is None:
            status = "undefined"
            break
        midpoint_coordinates = coordinates + 0.5 * step_size * velocity
        midpoint_velocity = velocity + 0.5 * step_size * acceleration
        midpoint_acceleration = _chart_acceleration(
            decoder,
            _chart_point(z0, U0, midpoint_coordinates),
            U0,
            midpoint_velocity,
            aggregate_sigma_max,
        )
        if midpoint_acceleration is None:
            status = "undefined"
            break
        coordinates = coordinates + step_size * midpoint_velocity
        velocity = velocity + step_size * midpoint_acceleration
        points.append(_chart_point(z0, U0, coordinates).detach())
        coordinate_history.append(coordinates.detach())
        velocity_history.append(velocity.detach())

    if status == "defined" and _thin_svd(
        _thin_decoder_jacobian(decoder, points[-1], U0),
        aggregate_sigma_max,
    ) is None:
        status = "undefined"

    result: dict[str, object] = {
        "intrinsic_status": status,
        "latents": torch.stack(points),
        "coordinates": torch.stack(coordinate_history),
        "velocities": torch.stack(velocity_history),
        "steps_per_quarter": steps_per_quarter,
    }
    if target is not None and len(points) > steps_per_quarter:
        result["target_residual"] = _normalized_decoded_mse(
            decoder,
            points[steps_per_quarter],
            target,
            z0,
        )
    else:
        result["target_residual"] = None
    return result


def _chart_acceleration(
    decoder: Callable[[Tensor], Tensor],
    latent: Tensor,
    U0: Tensor,
    coordinates: Tensor,
    aggregate_sigma_max: Tensor,
) -> Tensor | None:
    thin_jacobian = _thin_decoder_jacobian(decoder, latent, U0)
    decomposition = _thin_svd(thin_jacobian, aggregate_sigma_max)
    if decomposition is None:
        return None
    latent_velocity = (coordinates.reshape(-1, *([1] * (U0.ndim - 1))) * U0).sum(0)
    second = _decoder_second_directional(decoder, latent, latent_velocity)
    response = second.flatten().to(torch.float64) / math.sqrt(second.numel())
    return -(_pseudoinverse(decomposition) @ response).to(dtype=coordinates.dtype)


def _thin_decoder_jacobian(
    decoder: Callable[[Tensor], Tensor],
    latent: Tensor,
    U0: Tensor,
) -> Tensor:
    return _decoder_jvp(decoder, latent, U0).flatten(1).T.contiguous()


def _decoder_jvp(
    decoder: Callable[[Tensor], Tensor],
    latent: Tensor,
    directions: Tensor,
) -> Tensor:
    latent = latent.to(torch.float32)
    responses: list[Tensor] = []
    for start in range(0, directions.shape[0], _JVP_MICROBATCH):
        tangent = directions[start : start + _JVP_MICROBATCH].to(torch.float32)
        primals = latent.unsqueeze(0).repeat(
            tangent.shape[0],
            *([1] * latent.ndim),
        )
        _, response = torch.func.jvp(decoder, (primals,), (tangent,))
        responses.append(response)
    return torch.cat(responses)


def _decoder_second_directional(
    decoder: Callable[[Tensor], Tensor],
    latent: Tensor,
    direction: Tensor,
) -> Tensor:
    def one_point(point: Tensor) -> Tensor:
        return decoder(point.unsqueeze(0)).squeeze(0)

    def first_directional(point: Tensor) -> Tensor:
        return torch.func.jvp(one_point, (point,), (direction,))[1]

    return torch.func.jvp(first_directional, (latent,), (direction,))[1]


def _thin_svd(
    columns: Tensor,
    aggregate_sigma_max: Tensor,
) -> tuple[Tensor, Tensor, Tensor] | None:
    left, singular_values, right = torch.linalg.svd(
        _normalized_columns(columns),
        full_matrices=False,
    )
    floor = _NUMERIC_RANK_RTOL * aggregate_sigma_max.to(torch.float64)
    if not bool((singular_values > floor).all()):
        return None
    return left, singular_values, right


def _normalized_columns(columns: Tensor) -> Tensor:
    return columns.to(torch.float64) / math.sqrt(columns.shape[0])


def _pseudoinverse(decomposition: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    left, singular_values, right = decomposition
    return (right.T / singular_values) @ left.T


def _inverse_square_root(decomposition: tuple[Tensor, Tensor, Tensor]) -> Tensor:
    _, singular_values, right = decomposition
    return (right.T / singular_values) @ right


def _chart_point(z0: Tensor, U0: Tensor, coordinates: Tensor) -> Tensor:
    return z0 + (coordinates.reshape(-1, *([1] * (U0.ndim - 1))) * U0).sum(0)


def _normalized_decoded_mse(
    decoder: Callable[[Tensor], Tensor],
    prediction: Tensor,
    target: Tensor,
    origin: Tensor,
) -> Tensor:
    predicted_output = decoder(prediction.unsqueeze(0))
    target_output = decoder(target.unsqueeze(0))
    origin_output = decoder(origin.unsqueeze(0))
    denominator = (target_output - origin_output).square().mean().clamp_min(
        torch.finfo(torch.float32).tiny,
    )
    return (predicted_output - target_output).square().mean() / denominator


def _numeric_floor(singular_values: Tensor) -> Tensor:
    if singular_values.numel() == 0:
        return torch.zeros((), device=singular_values.device, dtype=torch.float64)
    return _NUMERIC_RANK_RTOL * singular_values[0]


def _orient_rows(rows: Tensor) -> Tensor:
    if rows.shape[0] == 0:
        return rows
    maxima = rows.abs().argmax(dim=1)
    signs = rows[torch.arange(rows.shape[0], device=rows.device), maxima].sign()
    return rows * torch.where(signs == 0, torch.ones_like(signs), signs)[:, None]


def _undefined_transport(U0: Tensor, initial_tangent: Tensor) -> dict[str, object]:
    rank = U0.shape[0]
    return {
        "intrinsic_status": "undefined",
        "coordinates": initial_tangent.detach().clone().unsqueeze(0),
        "frames": U0.new_empty((0, rank, rank)),
        "maps": U0.new_empty((0, rank, rank)),
        "singular_values": torch.empty(
            (0, rank), device=U0.device, dtype=torch.float64
        ),
        "return_tangent": initial_tangent.detach().clone(),
        "return_frame": U0.new_empty((0, 0)),
    }
