# Spec 0055: Kaggle Progress And Numerical Retry Contract

Status: locked / implementation-ready
Owner/workstream: Spec 0053 result-blind numerical preflight remediation
Last updated: 2026-09-10

## Purpose

Replace neither the failed Spec 0054 artifact nor its receipt. Define the
observability and numerical design required before a separately authorized,
new-slug retry can test Spec 0053 numerical viability. The failed `/1` used a
globally L2-normalized latent direction, so FP32 central differences changed a
typical coordinate by only about `6.25e-6` (JVP) or `1.25e-5` (HVP) and stopped
at a generic eager-autodiff gate before producing diagnostics.

This authorizes only local implementation and validation. A parent push still
needs the explicit remote-write authorization and guard below; it authorizes no
training, checkpoint alteration, sealed-data access, dataset write, scientific
comparison, or child continuation.

## Decision

Do not change the fixed latent shape `(16,32,32)`: one dense direction already
touches all 16,384 coordinates. The retry must use eight distinct, deterministic
dense Rademacher tangents whose coordinate RMS is one (`±1` per coordinate),
evaluated sequentially. Thus a fixed central step changes each coordinate by
the declared step rather than dividing that step by the square root of the
latent size.

For every tangent, use a distinct deterministic Rademacher cotangent for the
VJP identity. Keep FP32 forward/autodiff mechanics and move only detached
reductions to CPU FP64. The fixed primary central steps are `1e-3` (JVP) and
`2e-3` (HVP); `2e-3` and `4e-3` respectively are diagnostic-only companion
steps. The primary step alone decides acceptance. The retry must not choose the
lowest observed error after execution.

For JVP/HVP use the symmetric residual
`||a-b||₂ / max(||a||₂, ||b||₂, 1e-12)`. For VJP use the absolute bilinear
defect divided by `max(||Jv||₂||w||₂, ||v||₂||Jᵀw||₂, 1e-12)`, avoiding an
unstable denominator from a nearly cancelled inner product. Persist and print,
per check, the CPU-FP64 aggregate residual and the worst of the eight
projections. The gate requires both to meet the predeclared limit; a mean alone
is forbidden because it can hide an unstable direction. `p50` and `p95` are
diagnostics only.

## Result-Blind JSONL Progress

Remote stdout is a public result-blind output. The retry wrapper must expose one
controlled `emit_progress` helper, using
`print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)`, and
validate every event with the same output blindness guard before printing.
Every event has schema `spec0053.preflight_progress.v1`, a monotonic sequence,
a controlled event/stage identifier, and elapsed seconds.

Required events are `run_started`, `payload_ready`, `runtime_policy_ready`,
`contract_ready`, `fixtures_started`, `fixtures_complete`, `sample_ready`,
anonymous `branch_started`/`branch_complete`, workload start/complete, one
per-direction eager measurement before its gate, eager summary, compilation
start/complete, direction-ladder start/complete, output start/complete,
`run_complete`, and `failed`. Workload events may report only declared tensor
shapes, dtype, elapsed seconds, device/runtime version, and VRAM counters.
Eager events may report only anonymous branch label, direction index, fixed
step, named residuals/limits, aggregate/worst/p50/p95, and failed check IDs.

Events must never contain credentials, raw data, paths, input/checkpoint hashes,
selector contents, model identity/order, branch mapping, full contracts,
exception text, or traceback. The failure event carries only a controlled stage,
failure code, exception class, sequence, elapsed seconds, and safe GPU counters.
No raw `traceback.print_exc()` or `str(error)` is permitted in an uploadable
wrapper. Ruff T201 is disabled only for `kaggle/kernels/**` so these flushed,
controlled events can be printed; it remains enabled for other repository code.

## Required Retry Contract

The immutable contract is
`docs/data/spec0055_preflight_retry_contract.json`. Its fresh parent is exactly
`maximshtefan/eqvae-functional-geometry-preflight-02a08ab5`, its child identity
is reserved as `maximshtefan/eqvae-functional-geometry-preflight-02a08ab5-resume`,
and its one-shot guard is `spec0055_numerical_retry_authorized`. The parent is
private T4 with internet disabled and the same frozen, owner-qualified inputs.
The failed Spec 0054 `/1` receipt/log is negative evidence only; it cannot be a
predecessor or source for the new child.

## Verification And Blockers

Implementation must add CPU analytic tests for all eight distinct RMS-one
directions, symmetric JVP/HVP reduction, VJP scaling, aggregate-plus-worst
acceptance, and a failing-direction case that a mean-only gate would miss. It
must parse emitted JSONL, enforce the event allow-list/blindness guard, require
pre-gate residual events, and reject raw tracebacks. It must then run focused
tests, Ruff, BasedPyright, build/validate/check, `agent_preflight.sh`, and
`git diff --check`.

Fresh independent reviews must approve numerics, blindness/observability, and
provenance before the new contract becomes implementation-ready. A push requires
the new guard plus explicit current-turn user permission; no automatic retry or
wait loop is allowed.
