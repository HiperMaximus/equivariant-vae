# Decision 0004: SO(2) Gaussian-Ring Kernel Basis

Status: active

## Decision

Repo-owned steerable convolutions use Gaussian radial shells multiplied by real
circular harmonics. Legal pair-derived angular orders are selected by the
input/output irreducible representations. Spatial angular frequency `q > 0`
has zero support at the origin; same-frequency `q = 0` intertwiners remain
legal.

The frozen F0/F1 architecture uses the accepted `9-low` stem and `7-low`
remaining-kernel manifests with `q <= 2`. Basis samples are fixed buffers;
learned parameters are expansion coefficients; forward execution expands to
dense `conv2d` weights.

## Consequences

- F2 and higher-q profiles are excluded from the frozen comparison.
- Kernel support, ranks, conditioning and parameter counts are validated
  against the accepted oracle evidence.
- Fourier-Bessel or learned-radial alternatives require a separately scoped
  matched experiment.
