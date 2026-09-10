# Spec 0056: Frozen-Bundle Retry Preflight

Status: locked / implementation-ready
Owner/workstream: Spec 0053 result-blind preflight remediation
Last updated: 2026-09-10

## Purpose

Spec 0055 parent `maximshtefan/eqvae-functional-geometry-preflight-02a08ab5/1`
failed in `frozen_state` before a model or numerical workload. Its controlled
log/receipt are immutable negative evidence. The defect was the expected
`spec0045_vae_test_input.json` SHA-256: `b5d32...` in Spec 0055 instead of
the canonical `b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1`.
The latter is independently fixed by Specs 0046, 0047, 0050, 0051, and the
prior preflight which reached numerical execution.

This authorizes one new local package and one separately user-authorized,
private parent only: `maximshtefan/eqvae-functional-geometry-preflight-03a08ab5`.
It authorizes no training, checkpoint/data mutation, sealed access, scientific
comparison, dataset write, automatic retry, or child continuation.

## Fixed Design

The immutable machine contract is
`docs/data/spec0056_frozen_bundle_retry_contract.json`. It preserves all
Spec 0055 numerical and result-blind controls: eight deterministic dense
Rademacher directions, fixed primary/diagnostic steps, CPU-FP64
aggregate-plus-worst acceptance, and closed-schema flushed JSONL telemetry.

The wrapper must emit `frozen_bundle_ready` after exactly one canonical
bundle is found and before loading either model. The event contains no path,
hash, filename, model identity, or raw data. It permits the next failure to be
distinguished from a bundle lookup failure without weakening output blindness.

## Verification And Guard

The new guard is `spec0056_frozen_bundle_retry_authorized`. Before a push,
focused tests must prove the canonical bundle SHA is used, the pre-load event
is allow-listed and emitted, and the old `b5d32...` value is absent from
uploadable source. Run focused tests, Ruff, BasedPyright, build/validate/check,
`agent_preflight.sh`, and `git diff --check`; obtain fresh independent
numerical, observability, and provenance approval. Push only with explicit
current-turn user permission; do not wait automatically or launch the child.
