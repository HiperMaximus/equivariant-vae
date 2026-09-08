# Decision 0012: Kaggle Runtime Identity

Status: active

## Decision

Every GPU kernel installs or selects the exact Torch/CUDA stack declared by its
spec before importing project code. The run records Torch build, CUDA build,
driver, GPU identity/capability and compiler/backend versions.

Training and evaluation kernels may use different pinned stacks only when their
own accepted specs declare them. A performance-relevant stack change requires a
new bounded correctness and capacity check before expensive execution.

## Consequences

- Kaggle's preinstalled Torch is not implicit authority.
- Runtime installation is explicit, fail-closed and recorded.
- Existing accepted artifacts retain their recorded stack; no current run is
  floated to a newer release automatically.
