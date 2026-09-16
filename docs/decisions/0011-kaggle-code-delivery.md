# Decision 0011: Kaggle Code Delivery

Status: active

## Decision

Kaggle script kernels use one small direct code file that sparse-clones this
public repository and records the exact executed Git commit. The commit is
pushed immediately before the requested kernel version; repository payloads
are never embedded or regenerated per experiment.

Large checkpoints and binary data remain versioned Kaggle inputs referenced by
their exact owner-qualified locators.

## Consequences

- Repository source, not notebook JSON, is executable truth.
- `scripts/kaggle_kernel.sh` uploads only metadata and the direct code file.
- Internet access serves the declared public Git clone; mounted data locators
  remain independent of the authenticated actor.
