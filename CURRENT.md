# Current Repository Status

Last updated: 2026-07-22

## Read before working (agents keep skipping this)

Ground yourself via the landing sequence in `CLAUDE.md`, and **read the `AGENTS.md`
hard rules (1–31) — they are BINDING, not background.** Agents keep relapsing into
already-corrected traps because they skim them. The recurring ones:

- **Rule 30 — speed over declared don't-cares.** Judge the run on time-per-epoch +
  quality, NOT reproducibility or tail-completeness. Never add deterministic-algorithm
  / per-sample-seed / remainder-batch machinery "to be safe"; `drop_last=True`,
  `cudnn.benchmark=True`, fp16-first. See [[eqvae-speed-first-dont-cares]].
- **Rule 15 — keep the repo LEAN and LIVE.** DELETE non-current / historical /
  prose-only text outright; never banner-flag it, relabel it "historical", or append
  around it. Write compactly the first time. Git + superseded specs hold the past.
- **Rule 29 — verify the premise** before de-pinning a validator, relaxing a
  constraint, or deleting something that only "looks unused".
- **Rule 31 — every test docstring states its INTENT** (the invariant, why it matters,
  whether the expected value is a POLICY / a MEASURED cross-check / a DERIVED relationship,
  and what mutation it would catch). A test that cements a defect is worse than no test:
  when the fix lands it fails, and the next agent "fixes" the code to satisfy it.
- **Rule 22 — run the gate DETACHED via `setsid`**, capture the exit code, never pipe
  it into `tail`/`head`. It runs ~10 min (> the 600s tool cap); a naive `&` run is
  reaped mid-pytest (~63%, NOT a failure). See [[eqvae-gate-detach-and-exit-code]].

State of record for the active work: the **spec 0011 S17f body**
(`docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`) + the plan
memory `eqvae-reusable-runtime-mechanism-plan`. Keep this file and those current by
DELETING stale text, not appending.

## Current step

**Spec 0011 → S17f → Metrics part 2.** Commits **V** and **T** are DONE. V (`e51b262`)
aggregated validation on-device (one sync/view, additive `*_std` columns). T (`0ddc857`,
gate 604/1, 4 reviewers 0 confirmed) buffers the training per-step metrics on-device: the
step metric fields are 0-dim device tensors, the `_global_grad_norm`-family helpers return
tensors, and a persistent `_TrainStepMetricBuffer` index-writes each step with no host sync
and materializes the half-epoch window in one `.tolist()` — ~14 per-step syncs → ~0
(amp-off) / 1 (fp16 GradScaler floor); every per-step row kept, CSV/gate schema unchanged,
both step paths covered (eps telemetry stays host — CPU-computed, never a device sync). The
training buffer is **fp32** and is filled in place (no `torch.stack` temporary): it only
stores, never aggregates, so no accumulation error can build — **fp64 is reserved for the
Commit V validation accumulator**, which does sum across batches.
**A1 + B1 DONE (2026-07-22, gate 608/1).** `72ef19e` A1 — `wrap_fastpath_ddp` now applies
the dynamo config ITSELF, immediately before constructing DDP, with `dynamo` a REQUIRED
kwarg. DDP latches `optimize_ddp` at construction, so configuring it afterwards (what the
runner and executor did) silently left DDP on its C++ reducer: no `python_reducer`, zero
comm/compute overlap, no error. Made structural, not conventional. `278f1fd` B1 —
`PatchTrainingBatch` implements the `pin_memory()` hook torch dispatches on; without it
`pin_memory=True` was a silent no-op. B1 is a PREREQUISITE, not a win (see S17d below).

**NEXT = A2, and it gates the paid Kaggle run.** An FSQ-floor audit (2026-07-22, three
agents + verification) found the compiled-step search is hard-guarded to `amp_off_fp32`:
`runtime_selection_executor.py:3012-3019` and `real_data_runtime_pretest.py:1159-1166`
RAISE unless the policy is fp32 and hardcode `autocast_enabled=False`. The guard's own
docstring says why — "the closure hardcodes `autocast_enabled=False` and no GradScaler" —
i.e. AMP was never implemented in the compiled probe, so the axis was forbidden instead.
Measured on our own hardware: eager fp16 **27.38** samples/s (the committed plan, 30.4 h),
fp32+compile **18.01** (46.3 h), **fp16+compile 34.83 (23.9 h)**. So a compiled winner can
currently only be SLOWER than what we already run. A2 = wire autocast + a real GradScaler
through both compiled-step measurement branches (the runner already does this in
`_maybe_build_compiled_step`), delete the guard, add fp16 compiled policies to the grid.
**Two things A2 must NOT be shipped without** — both are spelled out in the spec's A2 bullet:
(1) **A6**, or the measurement is unselectable — `_compiled_row_stable` still rejects any row
with `graph_breaks != 0` and every real compiled row has `gb=1` (the DDPOptimizer bucket
split), so A2 alone measures fp16-compiled and then discards it; (2) the
`_efficiency_row_enumerable` **companion requirement** — an AMP policy is only enumerable at
`dual_batch_sizes` `[4,8,12]` while `efficiency_followup` runs `[12,48]`, so flipping the
compiled policy to fp16 SILENTLY drops bs48 unless you also provide an fp32-eager companion
at that batch. **Do all of this BEFORE the S17 generator run** or the paid run measures the
wrong space. Full ranked gap list (A2–A6, B3, B4, C-hash, D1/D2/D4) is the
`### S17f FSQ-floor gap plan` subsection of spec 0011 — the repo, not a memory, is the
contract.

Also remaining in S17f (all LOCAL gated commits): **Commit C** (shard `train_steps.csv`
into per-half-epoch `.csv.gz` — the ONE part-2 commit that changes a gate-reader contract),
compile-mode / `fullgraph`, DDP grad-overlap, and the blake2b retirement (which then
collapses the training batch to a bare tensor and DELETES the B1 hook). Then **S17d**
(bounded dataloader search — it has THREE blockers that must be fixed producer-first; the
spec records the order and why) + **S17e**. THEN Kaggle (user-driven, fresh window, exact
remote command + `KAGGLE_PUSH_CONFIRMED=1`): the S17 generator run → new compiled
`selected_runtime.json`, then **S19** (~30h + ~30 min staging, push-then-monitor), plus the
queued LR-finder.

NOTE (corrects an earlier claim in this file): the committed plan already runs **fp16 AMP**
(`autocast_dtype: float16`, `grad_scaler_enabled: true`). `amp_off_fp32` is the compiled-step
GRID candidate, not the current fallback — precision is not a standalone step, it is A2.

Each local step = ONE gated commit (detached gate) + a clean-context default-refute
adversarial review, both green, + explicit user approval; then roll this file, the spec,
and the plan memory forward by DELETING stale detail. Origin pushes are allowed on
request (never a bare `git push`).

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE using a repo-owned,
   compile-compatible implementation, with `escnn` as a reference.

## Update Rule

Update this file after meaningful shifts in active work, blockers, or next
steps, and before handing work back from a partial state. Each handoff update
should make clear:

- what changed;
- what is currently in progress;
- exactly where the agent left off;
- the next concrete action;
- active blockers or decisions needed;
- verification run and remaining failures.

Delete or replace stale information instead of appending contradictory history.

## VS Code Tasks

When opening this repo in VS Code, the local workflow tasks are:

- `Agent: preflight`
- `Paper: compile SIPAIM PDF`
- `Paper: Overleaf local check`
- `Python: quality`
