# 0004: SO(2) Kernel Basis First Run

Status: accepted
Date: 2026-06-11

## Decision

The first repo-owned continuous `SO(2)` steerable convolution implementation
uses Gaussian radial shells times real circular harmonics.

Locked first-run policy:

- maximum spatial angular frequency `L <= 2`;
- 5x5 kernels use Gaussian radial shell centers `[0, 1, 2]`;
- 7x7 kernels use Gaussian radial shell centers `[0, 1, 2, 3]`;
- default widths follow the useful `escnn` pattern: about `0.6` for interior
  rings, about `0.4` for the outer ring, and a tiny origin width;
- angular frequencies `m > 0` have zero support at the kernel center because
  the angular direction is undefined at `r = 0`;
- the center sample may still contain legal representation-theoretic
  intertwiners between compatible same-frequency input and output irreps;
- basis samples are precomputed as buffers, learned parameters are expansion
  coefficients, and the compiled forward path expands to dense `conv2d`.

Fourier-Bessel/Bessel radial bases remain a future fallback or ablation, not the
first real-run basis. Before using them, lock the disk radius, boundary
condition, radial order policy, and checks that sampled zeros do not remove too
much representation power on 5x5/7x7 grids.

## Rationale

For `SO(2)` steerable kernels, equivariance fixes the angular/intertwiner
structure, while the radial profile can be chosen. Gaussian shells are smooth,
local, easy to sample on small odd kernels, easy to cache as buffers, and match
the practical design used by `escnn`.

Pixel/delta rings are simpler but more grid-discrete and less smooth under
rotation. Bessel bases are mathematically clean, but they add radius, boundary,
zero-location, and conditioning choices that are unnecessary for the first
paper comparison.

## Consequences

Do not reopen the first-run SO(2) kernel basis unless the Gaussian-ring spike
fails an explicit equivariance, conditioning, or training-stability check.
Future Bessel work should be tracked as a separate spike/spec.
