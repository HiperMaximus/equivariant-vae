# Decision 0002: Normal VAE Baseline

Status: active

## Decision

The control is a normal denoising VAE with `mu`, `logvar` and the
reparameterization trick. It shares the `256x256` input, `(B,16,32,32)` latent,
stage schedule, optimizer budget, validation access and evaluation pipeline
with the continuous-`SO(2)` model.

The control uses ordinary convolution, GroupNorm and learned scalar gates, but
excludes FSQ/codebooks, PixelShuffle, final `tanh`, arbitrary pointwise channel
adapters and other operations without a fair steerable counterpart.

## Consequences

- The frozen 60,000-update checkpoint is the only comparison baseline.
- FSQ artifacts are runtime/macro-architecture reference material only.
- Architecture changes require a new matched two-model experiment.
