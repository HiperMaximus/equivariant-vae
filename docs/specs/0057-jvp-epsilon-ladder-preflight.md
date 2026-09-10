# Spec 0057: JVP Epsilon-Ladder Preflight

Status: locked / implementation-ready
Owner/workstream: Spec 0053 result-blind numerical remediation
Last updated: 2026-09-10

## Purpose

The sole Spec 0056 parent
`maximshtefan/eqvae-functional-geometry-preflight-03a08ab5/1` passed frozen
bundle lookup, strict model loading, and eight primary plus eight diagnostic
directions. It failed only the locked primary JVP finite-difference gate:
aggregate `0.0261526` and worst `0.0266058` exceeded `0.01`. Its twofold
diagnostic step lowered the aggregate to `0.0128344`, showing systematic
step-size sensitivity rather than an isolated projection.

This amendment authorizes one new local package and, only with current-turn
user authorization, one private parent:
`maximshtefan/eqvae-functional-geometry-preflight-04a08ab5`. It authorizes no
training, mutation, sealed access, model comparison, child launch, or automatic
retry.

## Fixed Diagnostic Protocol

The immutable machine contract is
`docs/data/spec0057_jvp_epsilon_ladder_contract.json`. The original primary
JVP test remains fixed at `epsilon=0.001`, with its existing `0.01`
aggregate-plus-worst acceptance gate. It is not relaxed and no diagnostic value
can pass or fail the parent.

For each of the same eight deterministic dense RMS-one Rademacher directions,
the wrapper additionally measures JVP finite differences at exactly `0.004`
and `0.008`, retaining HVP `epsilon=0.002`. It emits one flushed, closed-schema
result-blind JSONL row for every primary and diagnostic direction, followed by
separate aggregate/worst summaries for the two diagnostic JVP steps. Both
diagnostic steps run before the primary gate is asserted.

CPU-FP64 detached reductions, fixed tensor/model/input contracts, result
blindness, no TF32, deterministic algorithms, and the `frozen_bundle_ready`
event remain unchanged. The output must never expose paths, hashes, filenames,
model identities, branches' model mapping, or a scientific comparison.

## Interpretation And Guard

The two diagnostic results identify a numerically stable finite-difference
range; they do not select a new scientific model or amend acceptance after the
fact. A successor that changes the primary step or threshold requires another
locked amendment and explicit authorization.

Guard: `spec0057_jvp_epsilon_ladder_authorized`. Before a push run focused
tests, Ruff, BasedPyright, build/validate/check, shell syntax,
`agent_preflight.sh`, `git diff --check`, and fresh independent numerical,
observability, and provenance reviews. Push only with explicit current-turn
permission; do not wait automatically, retry, or launch the child.
