# Copyright 2026 HiperMaximus
"""Cross-rank DDP parameter-sync guard shared by the runner and the compile probe.

Correct DDP averages gradients during ``backward``, so after every optimizer step all
ranks must hold bit-identical parameters. Some misconfigurations (notably
``ddp_static_graph`` interactions, or a compiled step that drops the all-reduce)
silently skip that sync and let ranks drift into different models with no error.
Gathering a two-moment float64 parameter fingerprint (an unconditional, symmetric
collective, so it cannot itself desync) and raising on divergence turns a silent
divergence into an immediate, loud failure.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, cast

import torch
import torch.distributed as dist
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


def parameter_fingerprint(model: nn.Module) -> tuple[float, float]:
    """Return a two-moment float64 `(sum, sum-of-squares)` fingerprint of all params.

    Returns:
        The `(sum, sum-of-squares)` fingerprint over ``model.parameters()``.

    """
    parameter_sum = 0.0
    parameter_square_sum = 0.0
    for parameter in model.parameters():
        values = parameter.detach().to(dtype=torch.float64)
        parameter_sum += float(values.sum().item())
        parameter_square_sum += float(values.square().sum().item())
    return (parameter_sum, parameter_square_sum)


def assert_ddp_parameters_in_sync(model: nn.Module, *, world_size: int) -> None:
    """Fail fast if DDP ranks hold divergent parameters (gradients not synced).

    Gathers each rank's two-moment parameter fingerprint (a symmetric collective that
    cannot itself desync) and raises on every rank if any differs. Callers gate this to
    the first few optimizer steps of a process, where a systematic desync first appears,
    and confirm DDP is active/initialized before calling.

    Raises:
        RuntimeError: if any rank's parameter fingerprint differs.

    """
    gathered: list[object] = [None for _ in range(world_size)]
    all_gather_object = cast(
        "Callable[[list[object], object], None]",
        dist.all_gather_object,
    )
    all_gather_object(gathered, parameter_fingerprint(model))
    reference: tuple[float, float] | None = None
    for pair in gathered:
        if not isinstance(pair, tuple):
            continue
        fingerprint = cast("tuple[float, float]", pair)
        if reference is None:
            reference = fingerprint
        elif not _fingerprints_match(reference, fingerprint):
            message = (
                "DDP ranks hold divergent parameters after an optimizer step; "
                "gradient synchronization is not averaging across ranks (check the "
                "ddp_static_graph / torch.compile configuration)"
            )
            raise RuntimeError(message)


def parameter_sha256(model: nn.Module) -> str:
    """Hash every ordered parameter's metadata and exact bytes for untimed proof.

    Returns:
        A SHA-256 digest over parameter order, dtype, shape, and value bytes.

    Raises:
        RuntimeError: If any parameter contains a non-finite value.

    """
    digest = hashlib.sha256()
    for index, parameter in enumerate(model.parameters()):
        values = parameter.detach()
        if not bool(torch.isfinite(values).all().item()):
            message = "DDP parameter proof found a non-finite parameter"
            raise RuntimeError(message)
        digest.update(f"{index}:{values.dtype}:{tuple(values.shape)}:".encode())
        digest.update(
            values.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes(),
        )
    return digest.hexdigest()


def assert_ddp_parameters_exactly_in_sync(
    model: nn.Module,
    *,
    world_size: int,
) -> None:
    """Require complete exact parameter digests from every rank in an untimed proof.

    Raises:
        RuntimeError: If a rank digest is missing/malformed, parameters are non-finite,
            or exact ordered parameter bytes differ across ranks.

    """
    local_digest = parameter_sha256(model)
    gathered: list[object] = [None for _ in range(world_size)]
    all_gather_object = cast(
        "Callable[[list[object], object], None]",
        dist.all_gather_object,
    )
    all_gather_object(gathered, local_digest)
    if (
        len(gathered) != world_size
        or any(not isinstance(value, str) for value in gathered)
        or any(value != local_digest for value in gathered)
    ):
        message = (
            "DDP ranks hold divergent exact parameter bytes after an optimizer step"
        )
        raise RuntimeError(message)


def _fingerprints_match(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    """Return whether two parameter fingerprints agree, treating NaN as equal.

    Under correct DDP, synced parameters give bit-identical finite fingerprints, so
    exact equality is the right test. Identical NaN parameters across ranks are also
    in sync (a numerical blow-up is caught elsewhere), but ``nan != nan`` would
    wrongly flag them; NaN is matched to NaN so only a genuine divergence reports a
    desync.

    Returns:
        ``True`` if the two fingerprints agree moment-for-moment.

    """
    return _moment_matches(left[0], right[0]) and _moment_matches(left[1], right[1])


def _moment_matches(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return left == right


__all__ = [
    "assert_ddp_parameters_exactly_in_sync",
    "assert_ddp_parameters_in_sync",
    "parameter_fingerprint",
    "parameter_sha256",
]
