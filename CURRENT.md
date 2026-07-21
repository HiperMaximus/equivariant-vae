# Current Repository Status

Last updated: 2026-07-21

## Read before working (agents keep skipping this)

Ground yourself via the landing sequence in `CLAUDE.md`, and **read the `AGENTS.md`
hard rules (1–30) — they are BINDING, not background.** Agents keep relapsing into
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
- **Rule 22 — run the gate DETACHED via `setsid`**, capture the exit code, never pipe
  it into `tail`/`head`. It runs ~10 min (> the 600s tool cap); a naive `&` run is
  reaped mid-pytest (~63%, NOT a failure). See [[eqvae-gate-detach-and-exit-code]].

State of record for the active work: the **spec 0011 S17f body**
(`docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`) + the plan
memory `eqvae-reusable-runtime-mechanism-plan`. Keep this file and those current by
DELETING stale text, not appending.

## Current step

**Spec 0011 → S17f → Metrics part 2.** Commit **V** is DONE (`e51b262` — validation
aggregate + std: on-device fp64 accumulator, one sync/view, additive `*_std` columns,
means value-preserving). **NEXT = Commit T** — the training per-step on-device metric
buffer (full contract in the spec's "Commit T" bullet; `_SelectedRuntimeStepResult`
floats → device tensors, the `_global_grad_norm`-family helpers return tensors, per-step
no-sync index-writes materialized at the half-epoch flush, every per-step row kept, both
the eager `_run_train_step` and compiled `_run_compiled_train_step` paths — Commit V one
commit back is the working template). Then **Commit C** (per-half-epoch `.csv.gz` shards,
the one gate-reader-contract change).

Remaining S17f after T/C (all LOCAL gated commits): compile-mode / `fullgraph`, DDP
grad-overlap, precision → fp16 (its OWN step — `amp_off_fp32` was a bad agent default,
see [[eqvae-amp-off-was-bad-agent-default]]), and the blake2b retirement; then **S17d**
(bounded dataloader search — read its traps first; a de-pin was already attempted and
reverted) + **S17e**. THEN Kaggle (user-driven, fresh window, exact remote command +
`KAGGLE_PUSH_CONFIRMED=1`): the S17 generator run → new compiled `selected_runtime.json`,
then **S19** (~30h + ~30 min staging, push-then-monitor), plus the queued LR-finder.

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
