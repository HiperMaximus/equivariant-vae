# Decision 0002: Normal VAE Baseline

Status: active
Date: 2026-06-05

## Decision

The new non-equivariant baseline must be a normal denoising VAE with `mu`,
`logvar`, and the reparameterization trick.

The previous FSQ autoencoder runs are useful engineering evidence, but they are
not the comparable baseline for the paper.

This decision changes the bottleneck and objective, not the whole macro-
architecture. The first comparable baseline should keep the broad historical
FSQ/ResNet18-like residual encoder-decoder family, with spatial feature maps and
residual basic blocks. It must remove or replace non-translatable details such
as FSQ, PixelShuffle, 1x1 projections, pointwise channel adapters, and the
learned bottleneck scale. Standard GroupNorm remains allowed in the
non-equivariant Conv2d baseline for real-run stability, but the SO(2) path must
use field-aware normalization rather than raw GroupNorm over arbitrary channels.

Residual projection shortcuts are part of the architecture and must not be
implemented as naive shape adapters. When a residual branch changes spatial size
or channel count, the projection policy must have a documented route to the
future `SO(2)` model: fixed spatial resampling first, then an odd spatial
projection convolution. For encoder stage transitions, use the literature-backed
ResNet-D / anti-aliased ResNet pattern: branch-local fixed downsampling in both
the main branch and skip branch, replacing learned stride rather than
pre-downsampling before the residual split. Spec 0001 locks the fixed fieldwise
downsample operator to a 5x5 separable binomial low-pass filter followed by
decimation; it is selected from the future `SO(2)` side first and mirrored
exactly in the non-equivariant baseline. Resize/area downsampling is a future
fallback only if the locked operator fails a later SO(2) stage-transition test.

## Rationale

FSQ and quantized bottlenecks do not translate cleanly to the steerable
continuous `SO(2)` model. A fair comparison needs a baseline whose operations,
data contract, latent target, objective, and evaluation protocol can be mirrored
in the equivariant version.

## Consequences

- Use a Gaussian latent map and KL term.
- Preserve the ResNet-like residual encoder-decoder macro-architecture as the
  first comparable baseline unless a later decision explicitly supersedes it.
- Avoid vector quantization, codebooks, discrete indices, and tanh-bounded
  bottleneck assumptions in the comparable baseline.
- Avoid naive projection shortcuts; projection branches and branch-local
  downsample/upscale primitives must be specified, counted, tested, and mirrored
  by a future steerable counterpart.
- Keep the old FSQ results out of paper claims unless they are clearly labeled
  as historical engineering context or a separate ablation.
