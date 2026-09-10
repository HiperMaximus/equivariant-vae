# Matched VAE Architecture Contract

Status: implemented and frozen
Last updated: 2026-09-08

## Purpose

Define the architecture and fairness contract for the completed normal versus
continuous-`SO(2)` VAE comparison. Execution state and exact artifact identities
belong in `CURRENT.md`; detailed implementation gates belong in Specs 0001–0016.

## Shared Experiment Contract

| Item | Frozen value |
| --- | --- |
| Input | RGB `256x256`, normalized to `[-1,1]` |
| Latent | Gaussian posterior `mu, logvar` with shape `(B,16,32,32)` |
| Sampling | `z = mu + exp(0.5 * logvar) * eps` |
| Training budget | 60,000 committed optimizer updates per model |
| Data | Same pre-shuffled train/validation source and ordering contract |
| Loss | Matched denoising reconstruction objective plus KL schedule |
| Downsampling | Fixed anti-aliased blur plus decimation |
| Upsampling | Bilinear resize followed by convolution |
| Output | Raw zero-initialized RGB convolution; no final `tanh` |
| Evaluation | Shared fixed-25, reconstruction, rotation and downstream pipelines |

Both branches use matched stage shapes, residual/convolution counts, decoder
structure, optimizer access and validation access. The `SO(2)` learned parameter
count is no greater than the normal branch. Throughput, memory and fixed
resampling cost are reported rather than hidden.

## Normal VAE

The normal branch is the non-equivariant control. It uses ordinary `Conv2d`,
baseline GroupNorm and learned scalar gates while avoiding operations with no
clean steerable counterpart:

- no FSQ, vector quantization, codebooks or discrete bottleneck;
- no PixelShuffle or sub-pixel convolution;
- no final output `tanh`;
- no architecture-specific advantage in data, schedule or evaluation.

The broad encoder/decoder shape is ResNet-like, but the frozen implementation
is defined by Specs 0001, 0011 and its final checkpoint, not by notebook code or
the exploratory modules under `reference/`.

## Continuous-SO(2) VAE

The equivariant branch is repo-owned and specialized for planar continuous
rotations. `escnn` is a semantic/oracle reference, not a runtime dependency.

Frozen representation schedule:

- hidden fields use equal F0/F1 copy pairs `(16,16)`, `(24,24)`, `(32,32)`,
  `(48,48)`;
- the latent is `16F0`, preserving scalar Gaussian sampling;
- F2 was rejected by the numerical/support gate;
- the stem uses the accepted `9-low` kernel manifest and the remaining
  convolutions use `7-low`;
- analytic bases use Gaussian radial shells times real angular harmonics with
  zero center support for spatial angular frequencies `m > 0`;
- fixed basis expansion produces dense convolution weights compatible with the
  selected compiled runtime.

Field-aware operations:

- scalar/trivial fields use learned scalar gates comparable to the baseline;
- nontrivial F1 fields use radial gates with
  `r = sqrt(||v||**2 + eps)` and the configured FP16-safe epsilon;
- normalization respects irreducible field structure;
- fixed resampling acts fieldwise;
- arbitrary channel operations on geometric fields are forbidden.

## Objective And Output Domain

The VAE objective keeps reconstruction and KL components separately observable.
The decoder returns raw normalized RGB values. L1 is computed in that raw
domain; SSIM, PSNR and image artifacts use an explicit clamped projection.
Posterior monitoring includes `mu`, finite bounded `logvar`, KL, reconstruction
loss and beta schedule.

## Equivariance Evaluation

Separate these quantities:

- raw image transform/inverse-transform roundtrip error, which estimates the
  interpolation floor;
- reconstruction equivariance error under fixed sampled angles;
- posterior-`mu` transformation behavior;
- valid `logvar` behavior under the chosen representation;
- sampled-latent behavior using controlled or paired epsilon;
- decoded transformed-latent behavior;
- orbit smoothness, step uniformity and PCA planarity, which are distinct
  descriptive proxies.

The prior all-25, 360-angle local-linearity and traversal results are
superseded: the dense sweep mixed opposite rotation conventions at cardinal
angles. The separate exact-quarter control remains valid and does not favor
`SO(2)`. The corrected Spec 0050 sweep removes the seams but does not pass the
fixed regularity or harmonic gates, demonstrate clean internal-F1 behavior, or
find a transferable reduced action or local content--pose factorization.

## Fairness And Claim Gates

Before comparing branches, require:

1. identical data population and split;
2. identical input/latent shapes and training budget;
3. identical metric and qualitative pipelines;
4. reported parameter, memory, throughput and wall-clock differences;
5. explicit separation of validation, sealed test and exploratory diagnostics;
6. WSI-cluster uncertainty for WSI-level inferential claims;
7. no tuning, selection or automatic retry from sealed-test results.

The completed evidence meets these gates subject to the recorded normal
physical-update exception, the `SO(2)` 66/68 gate-health caveat, one training
trajectory per downstream branch and small sealed-test support.

## Downstream Contract

Frozen posterior-`mu` embeddings feed matched supervised heads:

- WSI diagnosis uses complete foreground bags and the fixed local-global MIL
  architecture;
- tissue classification uses nested balanced label budgets;
- model initialization, example order, optimizer and schedule are paired;
- development validation selects checkpoints; sealed test is consumed once;
- online training loss is optimization telemetry, not fixed-checkpoint train
  evaluation.

## Execution Boundary

The architecture transition is complete. Do not reopen the topology, runtime
search or training campaign during reporting. `CURRENT.md` owns the accepted
results and next authorized boundary; Specs 0017–0048 own data, supervised,
sealed-test, visualization and report details.
