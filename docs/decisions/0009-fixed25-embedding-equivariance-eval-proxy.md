# Decision 0009: Fixed-25 Rotation Evaluation

Status: active

## Decision

The fixed-25 rotated-input and transformed-embedding protocol is evaluation
only. It never contributes to the loss, augmentation policy or optimizer.

Both models use the same validation patches, angles, source hashes and
visualization pipeline. Exact quarter turns separate interpolation effects from
representation effects; continuous-angle orbits measure local smoothness,
step-size variation and PCA planarity as distinct descriptive proxies.

## Claim Boundary

- Fixed-25 evidence is validation evidence, not sealed test.
- PCA appearance is not a performance metric.
- Lower local-linearity ratio does not imply better traversal uniformity,
  planarity, reconstruction or downstream performance.
- The former all-25 dense result is superseded because its trajectory mixed
  opposite rotation conventions at cardinal angles. Do not reuse any dense
  smoothness, step-uniformity, PCA-orbit or pooled-F1 claim. Spec 0050 removed
  the seams but did not pass the fixed regularity/harmonic gates or demonstrate
  a clean internal-F1 transformation, shared reduced action or factorization.
  Exact-quarter controls remain valid and do not favor `SO(2)`.
