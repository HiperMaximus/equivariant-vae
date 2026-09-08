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
- The accepted all-25 result supports only greater one-degree local smoothness
  for `SO(2)`.
