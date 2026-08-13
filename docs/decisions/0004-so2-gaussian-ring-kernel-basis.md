# 0004: SO(2) Kernel Basis First Run

Status: accepted, revised
Date: 2026-06-11
Revised: 2026-08-12

## Decision

The first repo-owned continuous `SO(2)` steerable convolution implementation
uses Gaussian radial shells times real circular harmonics.

The field-frequency cap is two (`F0`, `F1`, `F2`); it is not a global cap on
spatial kernel harmonics. An `F_m -> F_n` block retains its legal pair-derived
orders `|n-m|` and `n+m`, through q=4 for `F2 -> F2`. Per-shell bandlimiting is
required to keep their sampled basis well-conditioned.

Spec 0012 owns one bounded FP64 search over 7x7/9x9 Gaussian centres, widths,
and per-shell q cutoffs. Coarse lattice-derived starts are refined by
deterministic multi-start COBYQA against a conditioning objective under direct
bounds/spacing constraints, then hard rank/condition and identically
configured `escnn` checks select a fixed manifest. The origin
is an exact q=0 impulse; q>0 is zero there, while same-frequency q=0 `I/J`
intertwiners remain legal. Basis samples are buffers, learned parameters are
expansion coefficients, and the compiled forward expands to dense `conv2d`.

The search selects only four global profiles: `7-low`, `7-full`, `9-low`, and
`9-full`. Their radii, widths, and shell cutoffs are shared by all convolution
positions assigned that profile. There is no encoder/decoder, stage, branch,
head, or per-layer radial search. Layer-specific basis sizes follow
deterministically from their input/output field multiplicities.

These alternatives belong only to the offline oracle. Once selected, the
training model hard-codes the one winning schedule and manifest; rejected
profiles do not become runtime configuration, adaptive behavior, or maintained
fallback implementations.

The measured oracle rejected F2 and selected the fixed F01 contingency. The
winning assignment is `9-low` at the stem and `7-low` everywhere else, with
the exact global radii, widths, and q masks stored in the Spec 0012 manifest.
The rejected `7-full`/`9-full` profiles are evidence only and cannot become
runtime branches. Fourier-Bessel and learnable radial profiles remain later
ablations, not first-run tuning parameters.

Capacity is locked separately in representation copies, not packed tensor
components. Each stage splits the baseline logical widths `[32,48,64,96]`
equally between F0 and F1 copies, giving copy pairs
`[(16,16),(24,24),(32,32),(48,48)]` and packed widths `[48,72,96,144]`. One F1
copy occupies two components but counts as one representation slot. The exact
analytic total is `1,180,035` learned parameters, below the `3,958,435` cap.

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

Use only the Gaussian manifest selected by Spec 0012's rank, conditioning, and
`escnn` reference gates. Future Bessel or learnable-radial work is a separate
ablation.
