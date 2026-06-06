# Decision 0001: Continuous SO(2) Scope

Status: active
Date: 2026-06-05

## Decision

The active equivariant target is continuous `SO(2)` steerability for planar
histopathology patches.

The intended implementation route is `escnn`, with a first target equivalent to:

```text
rot2dOnR2(N=-1, maximum_frequency=2)
```

## Rationale

The paper asks whether continuous rotation-equivariant structure helps a VAE
learn better histopathology patch representations. A discrete symmetry target
would answer a different question and would not match the current thesis/paper
goal.

## Consequences

- Baseline operations should be chosen because they can be mirrored in a
  continuous steerable model.
- Kernel sizes, field types, nonlinearities, normalization, upsampling, VAE
  statistics, and evaluation must be checked against continuous rotations.
- Optional reflection or other symmetry ablations must not distract from the
  first continuous `SO(2)` comparison.
