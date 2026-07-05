# Spec 0011: Reusable goal-derived runtime mechanism + compiled fast-path

Status: draft active
Implementation readiness: locked / implementation-ready (Phase 1); Kaggle phases gated on prior phases
Owner/workstream: selected-runtime speed + reusability
Last updated: 2026-07-05

## Purpose

Make the training runtime a **reusable, re-tunable mechanism** rather than a set
of frozen numbers, and use it to promote the compiled fast-path + bigger batch
into the first paper-promotable full run.

The batch size, learning rate, step schedule, speed recipe, and the
selected-runtime plan must all be **derived per (model × hardware)**, because the
repo's real target is the future equivariant model (the non-eq "translatable" VAE
is only the baseline). The equivariant model will have a different memory
profile, LR sensitivity, and compute, so its optimal batch/LR/schedule will
differ. Any value "just fixed" for the non-eq VAE (`batch=48`, `lr=0.001`, a
hand-written plan) is a landmine we would rip out later. See memory
`specs-encode-goals-not-frozen-numbers`.

**The shape (two-stage pipeline, run per model × hardware):**

1. **Efficiency search (pre-flight):** point the probe/sweep at *this* model on
   *this* hardware → it measures the throughput-optimal feasible batch, validates
   the winner compiled-DDP recipe, and **emits `selected_runtime.json`**.
2. **Full run:** consumes the generated plan → derives schedule + LR → trains.

The full-run config carries **no batch/LR number of its own**. The search is the
plan generator; the batch is a measured output; the schedule is formula-derived;
the LR is derived from a per-model reference; validators/gates check
**relationships**, not literals. When the equivariant model exists, re-run the
same machinery — no code surgery.

This spec supersedes the frozen-schedule parts of Spec 0009 (which pinned
`optimizer_updates_per_epoch=12500` / `max_train_steps=125000` etc.). B1 (commit
`119c8fd`) already un-froze the schedule in the config + runner target/save + the
`selected_runtime_full` kernel `_validate_full_config`; this spec finishes the job.

## Non-Goals

- No general model/representation/layer-schedule abstraction for a model that does
  not exist yet. The reuse seam is exactly **one tiny dict registry** keyed on
  `model.kind` plus relationship-based validators — nothing more.
- Not re-tuning the baseline's exact numbers to perfection. 48 vs 51, sqrt vs
  linear are transient for a model we will replace; the value is in the *mechanism*
  being clean and correct.
- No change to the fixed-25 equivariance protocol (Spec 0010) or the paper scope.

## The winner recipe (measured, compile probe v3, Kaggle dual-T4, 2026-07-05 PASS)

`optimize_ddp="ddp_optimizer"` (DDPOptimizer, **no** compiled_autograd) · whole-step
`torch.compile(step, dynamic=False, backend="inductor")` · `channels_last` ·
`DDP(gradient_as_bucket_view=True, find_unused_parameters=False, bucket_cap_mb=50,
static_graph=False)` · fused AdamW · drop the blake2b semantic-seed corruption on
the **train** fast path (keep it on validation/deterministic paths). The non-eq VAE
has 6 persistent buffers (`FixedBinomialLowpassDownsample2x.kernel`), so
`broadcast_buffers` matters — and its correct value is **model-specific** (must be
driven by a structural buffer check, not a hardcoded flag).

## Architecture / Workflow Contract

### Goal-derived relationships (the invariants every validator/gate enforces)

Let `P = REAL_TRAIN_PATCH_COUNT = 300000` (single-sourced constant; the design
located it near `benchmarking/synthetic_timing.py` — verify and reuse one source),
`G = global_batch = per_device_batch * world_size`, `E = epochs` (policy anchor).

- `global_batch == per_device_batch * world_size`
- `updates_per_epoch == floor(P / G)` — **floor**, because the train loader uses
  `drop_last=True` (see decision below). Single-sourced and threaded through the
  generator, runner, and gate.
- `target_train_steps == E * updates_per_epoch`
- `half_epoch_interval_steps == updates_per_epoch // 2` (floor; batch-independent)
- `save_every_steps == half_epoch_interval_steps`
- `resolved_beta_warmup_steps == updates_per_epoch` (one epoch)
- `scaled_learning_rate == reference_lr * (G / reference_global_batch) ** exponent`
  (`exponent`: sqrt→0.5, linear→1.0)
- **Boundary set (shared generator, used by BOTH runner and gate):**
  `sorted(set(range(half, target+1, half)) | {target})` — so the terminal
  checkpoint is always included even when `target` is not a multiple of `half`
  (at global 96, `target=31250` is not a multiple of `half=1562`).

### Fail-closed posture (must survive the de-pinning)

`E` stays an explicit **expected policy anchor** (a run cannot self-declare a tiny
self-consistent schedule). The remote gate keeps: strict
`optimizer_steps_completed == target`; exact `train_step_row_count == target *
world_size`; every step `1..target` present on every rank; cross-consistency
(`training_summary` schedule == plan; `full_summary` == `training_summary`). The
gate must **independently** assert `updates_per_epoch == floor(P / G)` using the
single-sourced `P` — not merely trust the sha256-verified plan (must-fix MF3).

### Model-decoupling seam (minimal)

`build_model(kind, *, norm_groups, ...)` in `src/eqvae/models/`, one entry today:
`'non_eq_vae_translatable' -> build_non_equivariant_vae`. Route **every** direct
`build_non_equivariant_vae()` call through it (runner `:972` and `:4342`, probe
`:598/:1041`, `runtime_selection_executor:1902`, `model_count`, `kaggle_smoke`,
`model_loss_train_step`, `cli/fixed25`, **and `training/debug.py` +
`benchmarking/real_data_runtime_pretest.py`** — MF5). Source the latent/eps shape
from the built model (`model.latent_channels` — must be **added** as an attribute,
MF5 — or a one-shot forward-probe) instead of importing the module constant
`LATENT_CHANNELS` into generic timing code (executor `:1804/:1983/:2006/:2029/:2057`).
Keep `_place_model`/`_maybe_wrap_ddp` typed to the base VAE / `nn.Module`. Eq-model
reuse = register `'eq_vae_so2' -> build_eq_vae` and set `model.kind` in its config.

### Per-model LR scaling

Reference lives in the **model's own config**, never the runner. Add
`optimizer.batch_lr_scaling = {reference_global_batch_size: 24, rule: "sqrt"}` to
`configs/spec0001/non_eq_vae_model_base.json` and reinterpret
`optimizer.learning_rate=0.0005` as the reference lr **at** that reference batch.
`optim.py`: `BatchLrScaling(reference_global_batch_size, rule="sqrt")` with an
`exponent` property + pure `scaled_learning_rate(*, reference_lr, scaling,
global_batch_size)`. Runner: `_optimizer_config(effective, *, global_batch_size)`
computes the scaled base lr and passes it to `SpecAdamWConfig.learning_rate` so all
group multipliers (decay 1.0 / no_decay 1.0 / gate_no_decay 0.5) ride on top
unchanged; `create_adamw_optimizer` math stays byte-identical. Record an
`optimizer_lr_scaling` provenance block in `training_summary`/`full_summary`; the
gate asserts the LR **relationship**. At global 24, `sqrt(24/24)=1` → 0.0005
unchanged (behavior-preserving).

### Plan provenance = one honest generator (option a)

Fold the compile-probe's feasible-batch sweep + winner-recipe measurement **into**
`runtime_selection_executor` so ONE generator, run on real dual-T4 data, measures
the batch + recipe and emits `selected_runtime.json` with the full linked-proof
graph that legitimately sets `full_run_eligible`. **Rejected:** probe-only (synthetic
tensors; `runtime_selection` is in the probe's BLOCKED_CLAIM_KEYS → dishonest) and
hand-assembly (non-regenerable; would fabricate the `runtime_proof` hashes the
parser cross-checks). The probe stays a non-promotable diagnostic.

## Config Contract

- `configs/spec0001/non_eq_vae_model_base.json`: add
  `optimizer.batch_lr_scaling = {reference_global_batch_size, rule}`; document that
  `optimizer.learning_rate` is now the **reference** lr at that batch.
- `SelectedRuntimePlan` schema (bump version): optional-with-eager-default recipe
  fields sourced from the measured winner row — `optimize_ddp`, `ddp_bucket_cap_mb`,
  `ddp_broadcast_buffers` (model-derived), `ddp_find_unused_parameters`,
  `fused_optimizer`, `compiled_autograd`, `reorder_compute_comm_overlap`, compile
  `backend`/`dynamic`, compile `scope` (add a `step`/whole-step value). Defaults
  reproduce the eager v5 recipe so old plans parse.
- Full-run config: carries **no** batch/LR/schedule numbers (B1 already removed the
  schedule; the batch comes from the plan; the LR reference lives in the model config).

## Implementation Plan (4 phases; each step = one gated commit)

Each step: `./scripts/python_quality.sh` green (ruff ALL + basedpyright strict +
pytest) + adversarial subagent review + explicit commit approval. Tests are updated
**within** each step. Phases 1 + 3 refactors are **behavior-preserving at batch 24**
(every derived value equals today's literal; recipe/DDP/compile/fused paths gated on
plan flags whose defaults reproduce the eager v5 plan). Only Phase 4 flips values.

### Phase 1 — Decouple + parametrize (local, behavior-preserving @ batch 24)

- **S1** Model-registry seam keyed on `model.kind` (+ MF5: include `debug.py`,
  `real_data_runtime_pretest.py`; add `model.latent_channels`).
- **S2** LR-scaling primitives in `optim.py` (pure additive).
- **S3** Wire sqrt LR into runner + model config at scale=1; record provenance.
- **S4** `fused` flag on `SpecAdamWConfig` (CUDA-guarded, default off; keep grouped
  param builder — do **not** adopt the probe's flat `model.parameters()`).
- **S5** Runner schedule validator → relationships; **floor** half-epoch (drop the
  even-guard). `updates_per_epoch = floor(P/G)` matching `drop_last=True`.
- **S6** Remote gate → relationships + LR + cross-consistency asserts; **gate
  self-anchors** to the dataset (`updates == floor(P/G)`, MF3); world_size from plan.
- **S7** Plan parser → relationship/structure checks (de-pin row_id identity, batch,
  recipe); **keep** the DDPOptimizer safety invariants + `runtime_proof` hash link +
  measured-snapshot cross-checks as the anti-fabrication backstop.
- **De-pin the kernel + packaging validators (MF1 — new; B1 left these mismatched):**
  `kaggle/kernels/selected_runtime_full/run_template.py` `_validate_selected_runtime`
  / `_validate_full_config`; `scripts/kaggle_kernel.sh:2136-2162` (the
  `expected_training` dict + the `FULL_TARGET_UPDATES = 125000` /
  `FULL_HALF_EPOCH_INTERVAL = 6250` required-text tokens + the `==12500` check);
  `tests/test_kaggle_embedded_kernel.py` asserts → derive from the plan. Regenerate
  `run.py` via the build script. **This is a latent break from B1 that only bites at
  full-kernel build — verify and fix here.**
- **Shared boundary generator + odd-batch end-to-end test (MF4):** one helper
  imported by runner + gate; a test that drives an odd-updates batch (e.g. global 96
  → 3125) end-to-end so the `31240`-vs-`31250` terminal/interval divergence is caught
  locally, before the ~30h run.
- **S9** Extract the shared fast-path recipe module (`training/fastpath_recipe.py`)
  from the probe (grouped fused-AdamW path, `_wrap_ddp`, `_apply_dynamo_config`) so
  probe and runner are bit-identical.

### Phase 2 — Generator emits the compiled plan

- **S8** Plan schema + generator payload recipe knobs (sourced from measured rows).
- **S10** Selector: whole-step compile as a first-class candidate, eligibility gated
  on the **relationship** (compiled AND strict settle-proof: post-settle
  `graph_break_count==0`, `recompile_count==0`, `settle_steps>=required`).
- **S11** Benchmark grid: add compiled whole-step + bigger-batch **candidate** rows;
  keep the **required** dual-gate rows eager-fp32 at batches that fit eager
  (`[4,8,12]`). Eager bs48 OOMs → bs48 is a compiled candidate, never a required
  eager gate row.
- **S12** [Kaggle] Fold the probe's single-GPU feasibility sweep (physical-free-VRAM
  gate, 1GB margin) + winner-recipe DDP timing into `runtime_selection_executor` as
  ONE generator that emits the full linked-proof graph.

### Phase 3 — Runner consumes the recipe (plan-gated, behavior-preserving off)

- **S13** DDP recipe wiring + fused enable via `fastpath_recipe.py`; `broadcast_buffers`
  by a **structural** model-agnostic rule (False only when every persistent buffer is
  a non-trainable rank-identical constant; else True). Keep `assert_ddp_parameters_in_sync`.
- **S14** Compiled step (`torch.compile(step, dynamic=False)` when `plan.compile_scope=='step'`)
  + train-only inline corruption (drop blake2b on train; keep it on
  validation/deterministic). **Keep `drop_last=True`** (user decision — 1 dropped
  step/epoch is meaningless data-wise); correctness comes from `updates_per_epoch =
  floor(P/G)` single-sourced (Phase 1), so the static batch dim needs no padding.
  Eager `_run_train_step` retained when compile is off.

### Phase 4 — Activate (values flip) + full run

- **S15** [Kaggle] Run the S12 generator on dual-T4 → new `selected_runtime.json`
  (winner = compiled bigger-batch row; new row_id). Re-point every row_id anchor.
  The reusable generator produces the artifact; de-pinned consumers accept it.
- **S16** Docs: de-pin Spec 0009 schedule passages to formulas (bs24 as a worked
  example only); add decision record `docs/decisions/0010-...` framing the mechanism
  as reusable across architectures; update CURRENT.md / specs README / open_follow_ups.
  Keep lean (net trim CURRENT.md).
- **S17** [Kaggle, ~30h + ~30min staging] First paper-promotable compiled full run;
  the relational gate certifies it against its own goal-derived schedule + LR
  relationship, failing closed on any truncation/gap.

## Decisions (resolved; not open forks)

- **Plan provenance** → (a) folded honest generator (only honest + reusable).
- **Batch source** → re-measured by the generator each run (never a consumed 48).
- **LR rule** → sqrt default (AdamW), per-model reference in the model config; user
  re-tunes `reference_lr`/`rule` per model. Baseline unchanged at scale 1.
- **`drop_last`** → keep `True`; make `updates_per_epoch = floor(P/G)` single-sourced.
  No-op for the baseline (24 and 96 both divide 300000); correct — not a landmine —
  for the eq model's non-dividing batch.
- **Odd updates_per_epoch** → floor half + the shared boundary generator (terminal in
  the final/best/latest lane). Do NOT constrain the selector to even-updates batches
  (that would distort the eq model's true optimal batch).
- **Model seam** → one dict registry keyed on `model.kind`; no larger abstraction.

## Acceptance Criteria

- Phases 1 + 3: `./scripts/python_quality.sh` green after each step; every derived
  schedule/LR value at the current v5 plan (global 24) is byte-identical to today's
  (12500 / 125000 / 6250 / 6250 / beta 12500 / lr 0.0005); the odd-batch e2e test
  passes at global 96 (3125 / 31250 / 1562, terminal 31250 covered).
- Phase 2/4: the folded generator emits a `selected_runtime.json` whose
  `full_run_eligible=true` is backed by real dual-T4 measured rows + linked proofs and
  a material speedup over the honest eager small-batch baseline.
- S17: strict `--verify-full-output` certifies the compiled run against its own
  derived schedule + LR relationship with zero launch blockers.

## Tests And Verification Commands

- `./scripts/python_quality.sh` (ruff ALL + basedpyright strict + pytest) per step.
- New odd-updates end-to-end test (global 96) in `tests/test_selected_runtime_full_run.py`.
- Kaggle: `--verify-full-output` gate on the downloaded run (S17).
- Local (CPU) authors + unit-tests compile/DDP/fused paths with compile off / fused
  off / contiguous; zero-graph-break / grad-sync / speedup are Kaggle-only observations.

## Known Risks / Adversarial Checks (from the design critique — verify each)

- **drop_last↔schedule (HIGH):** any `ceil` left in the schedule while the loader
  drops the tail → short run / DDP tail desync / gate coverage failure for a
  non-dividing batch. Single-source `floor` and grep for stray `ceil(patch/...)`.
- **Frozen-schedule homes are more than 6:** the in-kernel + packaging + embedded-test
  validators (MF1) must be de-pinned or the regenerated plan is rejected at Kaggle
  launch/packaging.
- **Gate self-anchor (MF3):** without an independent `updates==floor(P/G)` assert in
  the gate, a run could shrink its dataset coverage self-consistently.
- **Odd-batch boundary off-by-one (MF4):** terminal ≠ last interval at global 96;
  shared boundary generator + the e2e test must land before activation.
- **DDPOptimizer safety:** requires `find_unused_parameters=False`, `static_graph=False`;
  a mismatch silently drops the all_reduce. `broadcast_buffers=False` desyncs mutable
  persistent buffers with no error → the structural constant-buffer check is mandatory.
- **Honesty:** `full_run_eligible` must be earned by real-data linked proofs; report
  the gain as a combined recipe+batch improvement over the eager small-batch baseline
  (there is no same-batch eager baseline at bs48 — it OOMs), not as "1.4× compile".

## Open Questions

None blocking Phase 1. Confirm the single-source location of `REAL_TRAIN_PATCH_COUNT`
during S5/S6. The exact activation batch is a measured output of S12/S15, not a
decision.

## Related Files

- `GOAL.md`, `docs/repo_goal_and_requirements.md`, `docs/equivariant_vae_transition_plan.md`
- Spec 0009 (superseded schedule parts), Spec 0010 (fixed-25), `docs/decisions/README.md`
- Memory: `specs-encode-goals-not-frozen-numbers`, `eqvae-compiled-ddp-optimize-ddp`,
  `eqvae-fast-path-speed-priority`
- Design workflow output (17-step detail + critique):
  `.../54c70b5b-.../tasks/wvhy5430o.output` (session-local; distilled here)
