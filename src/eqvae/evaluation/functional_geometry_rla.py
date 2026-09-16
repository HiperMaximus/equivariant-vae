# Copyright 2026 HiperMaximus
# PyTorch's jvp/vjp stubs keep the 3-item has_aux return in their union.
# pyright: reportAssignmentType=false
"""Direct disposable JVP/VJP operators for decoder geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class LinearizedDecoder:
    """Matrix-free decoder operators whose autodiff graphs are per-call."""

    output: Tensor
    jvp_batch: Callable[[Tensor], Tensor]
    vjp_batch: Callable[[Tensor], Tensor]


def linearize_decoder(
    decoder: Callable[[Tensor], Tensor],
    latent: Tensor,
) -> LinearizedDecoder:
    """Build direct batched JVP/VJP operators with disposable graphs.

    Raises:
        ValueError: If the latent shape is invalid.

    """
    if latent.ndim < 2 or latent.shape[0] != 1:
        raise ValueError("decoder linearization requires one batched latent point")
    base_latent = latent.detach()
    with torch.no_grad():
        output = decoder(base_latent).detach()

    def jvp_batch(directions: Tensor) -> Tensor:
        _validate_directions(directions)
        primals = base_latent.repeat(
            directions.shape[0],
            *([1] * (base_latent.ndim - 1)),
        )
        with torch.enable_grad():
            primal_output, tangent = torch.func.jvp(
                decoder,
                (primals,),
                (directions,),
            )
        result = tangent.detach()
        del primal_output, tangent, primals
        if not bool(torch.isfinite(result).all()):
            raise ValueError("direct JVP responses must be finite")
        return result

    def vjp_batch(cotangents: Tensor) -> Tensor:
        _validate_directions(cotangents)
        primals = base_latent.repeat(
            cotangents.shape[0],
            *([1] * (base_latent.ndim - 1)),
        )
        with torch.enable_grad():
            primal_output, pullback = torch.func.vjp(decoder, primals)
            latent_rows = pullback(cotangents)[0]
        result = latent_rows.detach()
        del primal_output, pullback, latent_rows, primals
        if not bool(torch.isfinite(result).all()):
            raise ValueError("direct VJP responses must be finite")
        return result

    return LinearizedDecoder(
        output=output,
        jvp_batch=jvp_batch,
        vjp_batch=vjp_batch,
    )


def _validate_directions(directions: Tensor) -> None:
    if directions.ndim < 2 or directions.shape[0] <= 0:
        raise ValueError("directions require a nonempty leading batch dimension")
    if not directions.is_floating_point():
        raise TypeError("directions must use a floating dtype")
    if not bool(torch.isfinite(directions).all()):
        raise ValueError("directions must be finite")
