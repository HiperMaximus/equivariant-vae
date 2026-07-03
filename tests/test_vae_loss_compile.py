# Copyright 2026 HiperMaximus
"""SSIM torch.compile-cleanliness (fast-path port, step 1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from eqvae.metrics.reconstruction import ssim_per_image

if TYPE_CHECKING:
    from collections.abc import Callable


def test_ssim_per_image_traces_to_a_single_fullgraph() -> None:
    """ssim_per_image captures as one torch.compile graph (no .item() breaks)."""
    generator = torch.Generator().manual_seed(0)
    prediction = torch.rand((2, 3, 32, 32), generator=generator)
    target = torch.rand((2, 3, 32, 32), generator=generator)
    compiled_ssim = cast(
        "Callable[..., torch.Tensor]",
        torch.compile(  # pyright: ignore[reportUnknownMemberType]
            ssim_per_image,
            fullgraph=True,
            backend="eager",
        ),
    )

    result = compiled_ssim(prediction, target)

    torch.testing.assert_close(
        result,
        ssim_per_image(prediction, target),
        atol=0.0,
        rtol=0.0,
    )
