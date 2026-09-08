# Decision 0001: Continuous SO(2) Scope

Status: active

## Decision

The equivariant model targets planar continuous `SO(2)` steerability with a
repo-owned, compile-compatible implementation. Hidden fields use the selected
F0/F1 layout; `escnn` is an oracle/reference, not a runtime dependency.

Discrete rotation groups and `O(2)` reflections answer different questions and
are outside the frozen comparison.

## Consequences

- Arbitrary-angle rotations must be meaningful.
- Field-aware nonlinearities, normalization, resampling and latent statistics
  are required.
- F2 is not part of the frozen architecture.
- Any symmetry-scope change requires a new matched experiment.
