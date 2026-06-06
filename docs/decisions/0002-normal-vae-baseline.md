# Decision 0002: Normal VAE Baseline

Status: active
Date: 2026-06-05

## Decision

The new non-equivariant baseline must be a normal denoising VAE with `mu`,
`logvar`, and the reparameterization trick.

The previous FSQ autoencoder runs are useful engineering evidence, but they are
not the comparable baseline for the paper.

## Rationale

FSQ and quantized bottlenecks do not translate cleanly to the steerable
continuous `SO(2)` model. A fair comparison needs a baseline whose operations,
data contract, latent target, objective, and evaluation protocol can be mirrored
in the equivariant version.

## Consequences

- Use a Gaussian latent map and KL term.
- Avoid vector quantization, codebooks, discrete indices, and tanh-bounded
  bottleneck assumptions in the comparable baseline.
- Keep the old FSQ results out of paper claims unless they are clearly labeled
  as historical engineering context or a separate ablation.
