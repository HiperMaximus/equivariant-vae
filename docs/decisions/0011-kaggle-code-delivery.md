# Decision 0011: Kaggle Code Delivery

Status: active

## Decision

Kaggle script kernels receive repo code through a generated single-file
embedded payload. Generated wrappers bind a payload manifest and validate it
before importing `eqvae`.

Large checkpoints, manifests and binary datasets are versioned private input
datasets referenced by exact owner/slug/version and receipts; they are not
embedded in `run.py`.

## Consequences

- Repository source, not notebook JSON, is executable truth.
- Generated `run.py` and payload directories remain ignored.
- Internet settings are explicit per kernel and may serve only declared runtime
  installation needs.
- A public Git-based install route requires a separate delivery decision.
