# Spec 0012: Continuous SO(2) VAE Architecture

Status: implemented and frozen
Implementation readiness: closed; architecture is consumed by the final model
Owner/workstream: continuous-`SO(2)` VAE
Last updated: 2026-09-08

## Purpose

Define the selected repo-owned continuous-`SO(2)` architecture that matches the
normal VAE's spatial schedule while respecting field representations.

## Frozen Layout

| Resolution | F0 copies | F1 copies | Physical channels |
| ---: | ---: | ---: | ---: |
| 256 | 16 | 16 | 48 |
| 128 | 24 | 24 | 72 |
| 64 | 32 | 32 | 96 |
| 32 | 48 | 48 | 144 |

- RGB lift: `3F0 -> 16F0 + 16F1`.
- Hidden maps use the table's fixed equal-copy F0/F1 schedule.
- Posterior heads: `48F0 + 48F1 -> 16F0` for both `mu` and `logvar`.
- Decoder input: `16F0`.
- RGB projection: `16F0 + 16F1 -> 3F0`.
- Learned parameters: exactly `1,180,035`, below the normal VAE cap of
  `3,958,435`.

F2 is excluded. The architecture is not claimed to be the optimal field
allocation; it is the single frozen matched comparison model.

## Steerable Convolution

- Analytic basis: Gaussian radial shells times real circular harmonics.
- Origin: exact q=0 impulse; q>0 samples are zero at the center.
- Legal orders follow input/output irreducible-representation coupling.
- Selected profiles: `9-low` for the stem and `7-low` elsewhere.
- Retained spatial harmonics: q=0,1,2 only.
- Basis samples are non-learned buffers; independent learned expansion
  coefficients generate dense convolution weights.
- Runtime forward expands the fixed basis and calls dense `conv2d` through the
  accepted compiled path.

The committed oracle artifacts pin basis ranks, conditioning, normalization,
sign conventions and agreement with `escnn`. A failure of a legal q<=2 path is
a basis defect, not permission to alter the architecture.

## Field Operations

- F0 fields may use scalar bias, field-aware affine normalization and the
  shared learned scalar gate.
- F1 copies are two-component vectors. Their normalization and radial gate
  operate jointly on each copy.
- F1 gate radius is `sqrt(u**2 + w**2 + eps)` with `eps=1e-5` under the
  selected runtime.
- Fixed anti-aliased downsampling and bilinear upsampling act fieldwise.
- Residual projections preserve field type and apply fixed spatial resampling
  before steerable convolution.
- Arbitrary channel slicing, shuffling, GroupNorm or independent component
  nonlinearities are forbidden on nontrivial fields.

## VAE Semantics

- Posterior `mu` and `logvar` are scalar F0 fields.
- Reparameterization is
  `z = mu + exp(0.5 * logvar) * eps`.
- Decoder output is raw normalized RGB from a zero-initialized final
  convolution; no final `tanh` or clamp occurs in forward.
- L1 uses raw normalized output. SSIM, PSNR and saved images use explicit
  clamped image-domain projection.

## Equivariance Gates

Tests cover:

- basis/intertwiner identities and `escnn` agreement;
- scalar/F1 normalization and gating;
- residual blocks and stage transitions;
- downsample/upsample sampled-grid behavior;
- `mu`, `logvar`, paired-epsilon sampling and reconstruction rotation behavior;
- AMP, DDP, compile, parameter count and checkpoint compatibility.

Sampled-grid resampling error is reported separately from steerable-kernel
error. No test converts a qualitative PCA pattern into a performance claim.

## Boundary

Specs 0013–0016 own implementation, compiled mechanics, readiness and the
accepted 60,000-update checkpoint. The architecture is frozen; topology,
frequency schedule, basis profiles or parameter allocation cannot change during
evaluation or reporting.
