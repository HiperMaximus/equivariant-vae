# Spec 0009: Selected-Runtime Full-Run Lifecycle

Status: implemented compatibility contract
Implementation readiness: closed; no training launch authorized
Owner/workstream: normal-VAE full-run lifecycle
Last updated: 2026-09-08

## Purpose

Define the fail-closed checkpoint, resume and artifact lifecycle retained by
the selected-runtime full runner. Spec 0011 owns the executed 60,000-update
normal-VAE recipe and final evidence.

## Contract

- Resolve update counts and boundary cadence from the active config; do not
  hard-code a different schedule from this document.
- Count committed optimizer updates separately from physical attempts and AMP
  skips.
- Save model, optimizer, scaler, scheduler, progress and RNG state at atomic
  boundaries.
- A resume may consume only a fully verified boundary checkpoint with matching
  source/config/runtime identities.
- Preserve raw session outputs separately. Build a logical metric prefix only
  from committed rows at or below the verified checkpoint boundary.
- Primary-rank artifact writing is atomic; DDP workers must agree on committed
  progress and stop conditions.
- Validation, fixed-25 artifacts and checkpoints share the same logical update
  boundary.
- Final verification rejects missing files, mixed sessions, discontinuous
  metric prefixes, unrecorded AMP skips or checkpoint/hash mismatches.

## Boundaries

- This lifecycle is not permission to retrain or resume the frozen model.
- Its compatibility runner is not the current scientific result authority.
- `CURRENT.md` owns the final checkpoint and accepted exception.
- Spec 0010 owns fixed-25 evaluation semantics.
- Sealed test data is never available to this training lifecycle.

## Acceptance

The retained lifecycle remains covered by `tests/test_selected_runtime_full_run.py`
and the selected-runtime guard in `scripts/kaggle_kernel.sh`. Any code change to
that compatibility surface must keep these fail-closed invariants and pass:

```bash
.venv/bin/pytest -q tests/test_selected_runtime_full_run.py
.venv/bin/ruff check tests/test_selected_runtime_full_run.py
git diff --check
```
