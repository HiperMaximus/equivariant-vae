# Spec 0011: Reusable goal-derived runtime mechanism + compiled fast-path

Status: draft active — Phase 1 (S1–S10) + Phase 2 (S11–S13) + Phase 3 (S15/S16) + S14a/b/c DONE (committed local-only). Phase 4 STARTED: S17 decomposed into local sub-steps ahead of the paid run — **S17a DONE (recipe value-validators de-pinned to a coherence model, local)**; **S17b-1 DONE (parser identity made STRUCTURAL + snapshot batch/precision cross-consistency, local)**; **S17b-2 DONE (remote-output gate identity de-pinned to the loaded plan + both verifiers now validate the plan they derive from, local)**; **S17b-3 DONE (both `run_template.py` validators — `21b697f` — AND the debug `kaggle_kernel.sh` shell push guard — `c090d16` — delegate to `selected_runtime_plan_errors`, local)**; **S17c DONE (observation mirror + honest corruption/step label — `9f6d813`, local)**; **S17f item #1 (the `drop_last` unit-flip) DONE (`3b9aa42`, local)** + **S17f Transforms DONE (`2ce6a4c`, local — uint8 H2D + fold the uint8->float normalize into the compiled step; gate 575/1)** + **S17f cuDNN DONE (`a7feae4`, local — `cudnn.benchmark=True`/`deterministic=False` as a FIXED speed-first flag; gate 581/1)** + **S17f Full-validation DONE (`a6c6271`, local — the full run sweeps the WHOLE validation set every half-epoch, correcting the agent-set 20-batch cap; gate 583/1)** + **S17f RNG combined-step sub-commit 2 DONE (`5dde097`, local — the runner's eager + validation corruption move off blake2b onto the Philox `InlineStainCorruptor` via dedicated checkpoint-continued / re-seeded generators; corruption is now a FIXED property, not a selected axis; parser + label de-pinned; gate 586/1, 4 reviewers clean)** + **S17f Metrics part 1 DONE (`623f128`, local — the three per-parameter hot-step telemetry host-sync loops [`_global_grad_norm`/`_nonfinite_gradient_count`/`_parameter_update_norm`] vectorized to one on-device reduction each; value-preserving; gate 595/1, 5 reviewers clean)** + **S17f Metrics part 2 Commit V DONE (local — validation aggregate + std: on-device fp64 sum+sum_sq/view, one `.tolist()`/view, additive `*_std` columns, means value-preserving; gate 602/1, 6-lens review green)** + **S17f Metrics part 2 Commit T DONE (`0ddc857`, local — training per-step on-device metric buffer: step metric fields → device tensors + helpers return tensors, one `.tolist()`/half-epoch, ~14 syncs/step → ~0 (amp-off) / 1 (fp16), every per-step row kept, CSV/gate schema unchanged; gate 604/1, 4 reviewers 0 confirmed)**; NEXT local = the REST of S17f (compile-mode / `fullgraph`, DDP grad-overlap, Metrics part 2 Commit C [CSV shards — the one gate-reader-contract change], precision) + the blake2b retirement FOLLOW-UPS (SCOPE CORRECTION rule 29: blake2b was NOT dead after the runner-only sub-commit 2 — move the benchmark selection proof + debug/smoke/QA off blake2b, THEN delete it) + S17d (bounded dataloader search axis — read its traps before touching `_dataloader_errors`) + S17e (exact throughput-optimal batch search — producer follow-up); S17-Kaggle (row_id mint + dual-T4 run) + S19 + LR-finder stay Kaggle/user-driven
Implementation readiness: Phase 3 COMPLETE (local); S14a/S14b/S14c done + gated locally; S17a + S17b-1 done + gated locally (parser now ACCEPTS a self-consistent compiled plan — recipe AND structural identity/snapshot; identity is self-consistent so no Kaggle re-point is needed); compiled EXECUTION + the row_id mint are Kaggle observations; Kaggle phases S17-Kaggle/S19 gated (user-driven); LR-finder queued
Owner/workstream: selected-runtime speed + reusability
Last updated: 2026-07-22 (S17f Metrics part 2 Commit T DONE — `0ddc857`: the training per-step
metrics are a persistent on-device `_TrainStepMetricBuffer` — step metric fields → 0-dim device
tensors, the runner's `_global_grad_norm`/`_nonfinite_gradient_count`/`_parameter_update_norm`/
`_reconstruction_output_stats` return tensors, each step index-writes its 14 device scalars with no
host sync, one bulk `.tolist()` materializes the half-epoch window; ~14 syncs/step → ~0 (amp-off) / 1
(fp16 GradScaler floor); every per-step row kept, CSV/gate schema byte-identical, value-preserving,
both step paths; the 2 eps stats stay host (CPU-computed, never a device sync → buffer is 14 wide).
Gate 604/1, ruff+basedpyright clean; 4 clean-context default-refute reviewers → 0 confirmed; +2
mutation-proof buffer tests. NEXT = Commit C (CSV shards, the one gate-reader-contract change);
precision→fp16 stays a SEPARATE gated step [`amp_off_fp32` was a bad agent default, not the user's].
The per-step `(DONE — …)` tags in the body are the state of record.)

## Purpose

Make the training runtime a **reusable, re-tunable mechanism** rather than a set
of frozen numbers, and use it to promote the compiled fast-path + bigger batch
into the first paper-promotable full run.

**Speed-first, with two standing DON'T-CAREs (user, repeated — agents keep violating them).**
For every runtime/training choice, prefer SPEED: (1) exact bit/numerical **reproducibility is
NOT a goal** — small drift is fine, so `cudnn.benchmark=True`, non-deterministic algos, the
fastest RNG (drop per-sample blake2b seeding), fp16-first, and latest/beta torch features are
all in-bounds; (2) the dataset **TAIL does not matter** — `drop_last=True`, no remainder /
partial-batch logic (dropping a few thousand of ~300k patches is fine, and a fixed batch G is
what CUDA graphs need). Do NOT add reproducibility or tail-handling machinery "to be safe" —
only for a real correctness (NaN/divergence) reason. See the `eqvae-speed-first-dont-cares`
memory.

The batch size, learning rate, step schedule, speed recipe, and the
selected-runtime plan must all be **derived per (model × hardware)**, because the
repo's real target is the future equivariant model (the non-eq "translatable" VAE
is only the baseline). The equivariant model will have a different memory
profile, LR sensitivity, and compute, so its optimal batch/LR/schedule will
differ. Any value "just fixed" for the non-eq VAE (a hardcoded batch, an ad-hoc LR, a
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
`119c8fd`) un-froze **only** three things: the config schedule keys, the runner's
schedule **derivation** (`_target_train_steps`), and the `selected_runtime_full`
kernel `_validate_full_config`. It **did not** touch the runner's launch validator
`_validate_full_run_settings` (`selected_runtime_runner.py:5870-5916`) or its
`_FULL_*` module constants (`:131-134`) — which still literal-assert the whole
`12500 / 125000 / 6250 / 6250` schedule *plus* `save_every==6250` and
`beta_warmup==12500` — nor the plan parser `_launch_errors`
(`selected_runtime.py:409`), the `kaggle_kernel.sh` packaging validator, the remote
gate `REMOTE_FULL_*` constants, or the generator's `ceil`. B1's own commit message
notes "the still-pinned `_validate_full_run_settings` and remote gate keep passing".
This spec finishes the job — and note the `kaggle_kernel.sh` full-kernel build is
**broken today** (MF1): B1 stripped the config keys the shell validator still
requires, so `preflight-selected-runtime-full` and push fail-closed right now.

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
static_graph=False)` · fused AdamW · corruption is the vectorized inline
`InlineStainCorruptor` on both the train and validation runtime paths (blake2b retired
from the runner, RNG swap `5dde097`). The non-eq VAE
has 6 persistent buffers (`FixedBinomialLowpassDownsample2x.kernel`), so
`broadcast_buffers` matters — and its correct value is **model-specific** (must be
driven by a structural buffer check, not a hardcoded flag).

## Architecture / Workflow Contract

### Goal-derived relationships (the invariants every validator/gate enforces)

Let `P = REAL_TRAIN_PATCH_COUNT = 300000`, single-sourced (S6a `f154a84`) as the
canonical constant in `data/roots.py`; the gate anchor (MF2) reads THAT, never the
plan's number, never `data.real_train_patch_count`. `G = global_batch =
per_device_batch * world_size`, `E = epochs` (policy anchor).

- `global_batch == per_device_batch * world_size`
- `updates_per_epoch == floor(P / G)` — **floor**, matching the train-loader
  `drop_last=True` (S16 `3298a57`; the projection-record unit-flip is S17f `3b9aa42`).
  The floor is single-sourced via `benchmarking/schedule.py` `training_steps_per_epoch`
  (S5b `533f554`); the tiny-selector `ceil` at `runner:627` is unrelated and stays.
- `target_train_steps == E * updates_per_epoch`
- `half_epoch_interval_steps == updates_per_epoch // 2` (floor; batch-independent)
- `save_every_steps == half_epoch_interval_steps`
- `resolved_beta_warmup_steps == updates_per_epoch` — holds **only because
  `warmup_fraction (0.1) * epochs (10) == 1`**; the code derives it as
  `ceil(warmup_fraction * target_train_steps)` (`losses/vae.py:191`), which equals
  `updates_per_epoch` under floor for any batch. Any validator must assert the real
  formula `ceil(fraction*target)` (or assert `fraction*epochs==1`), not a raw
  `== updates_per_epoch` identity a future per-model config could silently break.
- `scaled_learning_rate == reference_lr * (G / reference_global_batch) ** exponent`
  (`exponent`: sqrt→0.5, linear→1.0)
- **Boundary set (shared generator, drives BOTH runner PRODUCERS and gate/runner
  CONSUMERS):** `sorted(set(range(half, target+1, half)) | {target})`. **Critical
  (MF3):** the terminal is NOT dropped today — `final.pt` always saves it, and the
  runner's modulo producers + range consumers already agree on the last grid step
  (31240 at global 96). Applying `| {target}` to the CONSUMERS only, while the
  PRODUCERS stay modulo (`% save_every` `:2872`, `% half` validation `:3969`), would
  make the gate demand a `step_031250.pt`/validation row the runner never writes →
  false-reject a valid ~30h run. Per the *validate-the-terminal* decision, the shared
  set must drive BOTH sides: convert the runner producers to set-membership so the
  terminal is genuinely checkpointed, validated, and best-selection-eligible.

### Fail-closed posture (must survive the de-pinning)

`E` stays an explicit **expected policy anchor** (a run cannot self-declare a tiny
self-consistent schedule). The remote gate keeps: strict
`optimizer_steps_completed == target`; exact `train_step_row_count == target *
world_size`; every step `1..target` present on every rank. It must **independently**
assert `updates_per_epoch == floor(P / G)` using the immutable single-sourced `P`
and `plan.global_batch` — not merely trust the sha256-verified plan (must-fix MF2).
Two additions the current gate lacks (net-new, not conversions): (a) a **direct**
cross-consistency assert `training_summary.schedule == plan.schedule` and
`full_summary.schedule == training_summary.schedule` (today they are only compared to
the shared literal, so removing the literal removes the glue; also add
`optimizer_updates_per_epoch` to `_application_mismatches`); (b) `world_size` sourced
**from the plan** for the expected-rank set / row count / coverage math
(`gate:877-881` hardcodes `REMOTE_FULL_WORLD_SIZE=2` — convert it, keep only a `>=1`
floor). The LR-relationship assert (S6) is likewise net-new and depends on S3
recording `optimizer_lr_scaling` provenance first.

### Model-decoupling seam (minimal)

`build_model(kind, *, model_config)` in `src/eqvae/models/`, one entry today:
`'non_eq_vae_translatable' -> build_non_equivariant_vae`. **Use opaque per-kind
kwargs** (unpack kind-specific args from the model-config block) — do NOT promote
`norm_groups` to the universal signature (R2): `norm_groups` is a GroupNorm-only
concept the field-aware eq model has no use for, so baking it in is itself a non-eq
coupling. Route **every** direct `build_non_equivariant_vae()` call through the
registry (all 12 verified sites: runner `:972` and `:4342`, probe `:598/:1041`,
`runtime_selection_executor:1902`, `model_count:304`, `kaggle_smoke:142`,
`model_loss_train_step:149`, `cli/fixed25:137`, `training/debug.py:310`, **and both**
`real_data_runtime_pretest.py:856` **and** `:3571` — MF4).

**Latent channels (R1 — the coupling that actually bites).** Add
`model.latent_channels` as an attribute and source the eps/latent shape from the
**built model** — not only in the executor timing code
(`:1804/:1983/:2006/:2029/:2057`) but critically **in the runner**, which is the
machinery the eq model re-runs: `_train_eps` (`:5669`), `_zero_eps` (`:5606`), and
`_write_reconstruction` (`:4582`) all build eps as `(B, LATENT_CHANNELS, H, W)` from
the imported non-eq constant (`:92`). Leaving the runner coupled means the first eq
run either crashes in `reparameterize` (shape mismatch, `non_equivariant_vae.py:371`)
or silently reads the wrong module — the exact frozen-number landmine this spec
exists to kill. Route (or explicitly scope as non-eq-specific, with reason) the other
constant sites too: `debug.py:1665/1707`, `model_loss_train_step.py:238/256`,
`real_data_runtime_pretest.py:897/918/941/3609`, `kaggle_smoke.py:343`. **NOT a
coupling:** the eps *distribution* (per-channel iid `N(0,1)`) is already isotropic
(cov `= I`) and invariant under orthogonal irrep representations, so it does NOT break
sampled-`z` equivariance — only the shape/source is coupled, not the noise; do not
"fix" the distribution.

`_place_model`/`_maybe_wrap_ddp` are currently typed to concrete `NonEquivariantVAE`
(`:2686`, `:2702`) with five `cast("NonEquivariantVAE", …).forward(eps=…)` sites
(`:3872/:4118/:4591/:3347/:3413`) — **retype** them to `nn.Module` / the registry
return type (a change, not a "keep") and document the required `forward(…, eps=)`
contract in the registry so the eq model's forward API is explicit. **channels_last +
whole-step compile (R3):** both are applied kind-agnostically (`runner:2690`) but an
`escnn`-style model wraps `GeometricTensor` / basis-expansion coefficients, so
`channels_last` may be a silent no-op and whole-step compile may graph-break. The plan
flags already allow `memory_format`/`compile_scope` to be eager per-model — so this is
a **re-measurement obligation** for the eq model, not necessarily code surgery; name
it as a risk. Eq-model reuse = register `'eq_vae_so2' -> build_eq_vae`, set
`model.kind` + `latent_channels` in its config.

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

- **S1** Model-registry seam keyed on `model.kind`, opaque per-kind kwargs (+ MF4:
  route all 12 build sites incl. `debug.py`, both `real_data_runtime_pretest.py`
  sites; add `model.latent_channels` AND source eps from the model **in the runner**
  `_train_eps`/`_zero_eps`/`_write_reconstruction`, not only the executor; retype
  `_place_model`/`_maybe_wrap_ddp`/forward casts to `nn.Module`).
- **S2** LR-scaling primitives in `optim.py` (pure additive).
- **S3** Wire sqrt LR into runner + model config at scale=1; record provenance.
- **S4** `fused` flag on `SpecAdamWConfig` (CUDA-guarded, default off; keep grouped
  param builder — do **not** adopt the probe's flat `model.parameters()`).
- **S5** Runner schedule validator → relationships; **floor** half-epoch (drop the
  even-guard `:1729`). Explicitly re-target `_validate_full_run_settings:5870-5916`
  (still literal-asserts the whole schedule — this is the primary `src/` home B1 left
  pinned), the `_FULL_*` module constants `:131-134`, the `save_every==6250` guard
  `:5906`, and the `beta_warmup==12500` guard `:5910` — keep `requested_epochs==E` and
  `validation_batches_per_view` as policy anchors. `updates_per_epoch = floor(P/G)`
  single-sourced; convert the four generator `ceil` sites + the config
  `schedule_derivation` docstring (`ceil`→`floor`). De-pin the existing literal test
  fixtures (`test_selected_runtime_full_run.py:45-48`), keeping bs24 as one example.
- **S6** Remote gate → relationships + LR + cross-consistency asserts; **gate
  self-anchors** to the dataset (`updates == floor(P/G)` against the immutable
  single-sourced `P`, MF2). Convert the `REMOTE_FULL_*` literals (`gate:137-140`) and
  the `REMOTE_FULL_WORLD_SIZE=2` coverage math (`:877-881` → `plan.world_size`), route
  the two boundary consumers (`_full_expected_interval_checkpoint_names` `:819`,
  `_remote_full_validation_blockers` `:947`) through the shared generator, and ADD the
  direct `training_summary==plan==full_summary` cross-consistency asserts + the
  LR-relationship assert (both net-new; LR needs S3 provenance first).
- **S7** Plan parser → relationship/structure checks (de-pin the `_launch_errors`
  `per_device_batch_size==12` / `global_batch_size==24` / `optimizer_updates_per_epoch==12500`
  literals → `global==per_device*world_size` and `updates==floor(P/G)` via the
  single-sourced `training_steps_per_epoch`; keep the hardware/policy anchors
  `accelerator_mode`/`machine_shape`/`world_size`/`nproc_per_node`/`gradient_accumulation_steps`
  pinned; preserve every error-name identifier; fail-closed on malformed/missing/typed
  fields — no `ZeroDivisionError`). The DDPOptimizer safety invariants do **not** exist
  yet (schema has only `ddp_static_graph`) — this is an **ADD**, not a keep: reject any
  plan with `optimize_ddp=='ddp_optimizer'` paired with **`compiled_autograd`**,
  `static_graph`, or `find_unused_parameters` True. Per memory
  `eqvae-compiled-ddp-optimize-ddp` and the measured winner `_DDP_OPTIMIZER_SPEC` (which
  uses `compiled_autograd=False`), `compiled_autograd=True` is the **silent** all_reduce
  drop (traced backward → C++ reducer hooks never fire → independent replicas);
  `static_graph=True` is a **loud** dynamo #93672 conflict; `find_unused_parameters=True`
  is incompatible with the bucket split. Read the recipe knobs defensively across the
  carrier blocks (their exact home is not frozen pre-Phase-2). Keep the `runtime_proof`
  hash link + measured-snapshot cross-checks + the 8-step `assert_ddp_parameters_in_sync`
  runtime guard as the anti-fabrication/safety backstop.
- **S8 (DONE — this commit) De-pin the shell packaging validators (MF1 — BROKEN
  TODAY):** scoped to `scripts/kaggle_kernel.sh` + `tests/test_kaggle_embedded_kernel.py`
  (Option A; `run_template.py` deferred to S8b). The *only* actually-broken surface was
  the FULL guard's `expected_training` config loop (`:2136-2148`): B1 stripped the four
  schedule keys the loop still required (`training.get()` → `None`), failing
  `preflight-selected-runtime-full` **and** push. **Correction to the prior handoff:**
  `run_template.py` `_validate_baseline_selected_runtime` was **NOT** broken — it reads
  the untouched `selected_runtime.json` (still 12/24/12500) and passed; the memory/handoff
  premise was wrong. The de-pin imports the single-sourced
  `training_steps_per_epoch`/`REAL_TRAIN_PATCH_COUNT` into the PYFULLPAYLOAD heredoc (via
  `PYTHONPATH=src "$python_bin"`, the launcher line 2050 already uses), drops the 4
  schedule keys + adds a fail-closed anti-re-freeze guard, derives the `selected_runtime`
  updates check + the `FULL_TARGET_UPDATES`/`FULL_HALF_EPOCH_INTERVAL` tokens, and converts
  the debug guard's `12/24` pins to `global==per_device*world_size`. Byte-identical `run.py`
  @ batch 24; every error-message string preserved; fail-closed on malformed/missing/typed
  fields. Tests extract the **real** guard heredoc from the script (no drift) and exercise
  it: pass on a fresh build + three fail-closed rejections (re-freeze, off-derivation
  updates, off-derivation `run.py` token).
- **S8b (DONE — this commit) De-pin `run_template.py`; unlock a non-24 full batch.** The
  `FULL_TARGET_UPDATES`/`FULL_HALF_EPOCH_INTERVAL` constants stay baked literals in `run.py`
  (the S8 shell token check greps them) but are now **build-derived**: the kernel builder
  regex-rewrites them from `floor(REAL_TRAIN_PATCH_COUNT / plan.global_batch)` × epochs.
  Two deviations from the draft plan, both forced by the code: (1) a **`string.Template`
  bare-`$` placeholder would break `run_template.py` as valid Python** (it is ruff-linted;
  the existing `$` placeholders are all inside string literals) — so the builder uses an
  anchored **regex substitution** (`FULL_TARGET_UPDATES_PATTERN`), fail-closed if either
  constant is not rewritten exactly once; (2) the builder imports
  `eqvae.benchmarking.schedule` + `eqvae.data.roots` NORMALLY to reuse the single source
  (never re-inlines `P`), and `_assert_eqvae_is_repo_root` fails the build closed if the
  imported `eqvae` is not the tree the payload ships. **SUPERSEDED 2026-07-15:** this step
  originally ran the builder under bare `python3` (no torch, no `eqvae`) and therefore
  loaded the two leaves BY FILE PATH via `_load_leaf_attr`. That is gone: `eqvae` is now
  editable-installed and `scripts/kaggle_kernel.sh` builds via `build_kernel_py()` on the
  venv interpreter. The "torch-less build" was never a requirement — only a consequence of
  picking the system interpreter. `_validate_baseline_selected_runtime` de-pinned to a
  runtime relationship (`global == per_device·world_size`; `updates == training_steps_per_epoch(...)`
  imported after `sys.path.insert`), so the now-unused `FULL_UPDATES_PER_EPOCH` constant was
  **deleted**. `FULL_EPOCHS = 10` stays (goal). The S8 shell token check needed **no change**
  (already derived). Byte-identical `run.py` @ batch 24; the import-artifact validators
  consume the build-substituted constants unchanged. Tests: import the builder + template
  modules to exercise the non-24 path — `_derive_full_schedule` @24 → 12500/125000/6250,
  `_apply_full_schedule_substitution` @96 → 3125/31250/1562 (+ fail-closed), and the
  de-pinned validator accepts `96 == 48·2` / rejects off-product + off-updates.
  Post-review hardenings folded in: the builder's substitution gate fires on **either**
  schedule pattern (a lone surviving constant then trips the fail-closed exactly-once
  assert); `optimizer_updates_per_epoch` is int-strict in both the run.py validator and
  the shell guard (mirroring the sibling fields); and a non-24 end-to-end
  `_derive_full_schedule` test (global-batch 96 → 3125/31250/1562). A dedicated
  torch-free-build test was **deliberately skipped** (torch is pre-imported in the pytest
  venv, so a robust guard needs a subprocess — low value); the bare-`python3` build path is
  instead exercised by the real `preflight-selected-runtime-full`.
- **S9 (DONE — this commit) Shared boundary generator + odd-batch test (MF3):** one
  helper (`boundary_steps`, in the stdlib-only `schedule.py` leaf) imported by ALL eight
  boundary sites — the runner PRODUCERS (interval checkpoint `% save_every` `:2904`,
  scheduled validation `% half` `:4009`) converted to set-membership, AND the CONSUMERS
  (runner `:2256/2295/2332/5387`, gate `:903/1130`). Per the *validate-the-terminal*
  decision, `| {target}` makes the terminal a real boundary on **both** sides
  (written + validated + best-selection-eligible) — never consumer-only (that
  false-rejects). Byte-identical @ batch 24 (`half=6250,target=125000`: `∪{target}` is a
  no-op, so the whole existing suite still passes). The literal global-96 e2e test is
  **not CPU-runnable** (full mode forbids capping below target without `--dry-run`,
  requires `completed_steps==target`, and `world_size=2` needs CUDA+nccl). Instead: (a) a
  pure-function unit test of the shared generator (`half=1562,target=31250 →` grid ends
  `31240`, terminal `31250` force-included); plus (b) a **tiny** odd full config CPU run
  (`epochs=1, updates_per_epoch=5 → half=2, target=5`, terminal off-grid) reproducing
  the topology in 5 steps single-process. World_size=2 stays a Kaggle-only observation.
  Post-review hardening folded in: an adversarial jury flagged the consumer tests as
  **vacuous** (they only ran on-grid schedules, so a consumer-only regression would pass
  silently) — closed with three off-grid consumer tests (runner schedule-complete +
  resume-prefix, gate interval-checkpoint names, gate validation-CSV blockers), each
  **mutation-proven**: reverting either the runner Site-D consumer or the gate
  checkpoint-names makes its guarding test fail.
- **S10 (DONE — this commit) Extract the shared fast-path recipe module
  (`training/fastpath_recipe.py`) from the probe.** One source for the three recipe
  components — `build_fastpath_optimizer` (grouped, CUDA-gated fused via
  `create_adamw_optimizer`), `apply_fastpath_dynamo_config`, `wrap_fastpath_ddp` — taking
  plain scalar knobs (source-agnostic, so both the probe's `_RecipeSpec` and the future
  runner's `SelectedRuntimePlan` drive them, never depending on each other's types). The
  probe delegates through thin adapters (`_build_optimizer`/`_apply_dynamo_config`/`_wrap_ddp`).
  LOAD-BEARING FIX: the probe's fused optimizer was a bespoke FLAT ungrouped
  `torch.optim.AdamW(model.parameters(), fused=True)` — it weight-decayed norms/biases/gates
  and dropped the 0.5x gate LR; it now uses the grouped path, matching the runner (which
  already builds via `create_adamw_optimizer`). Non-fused path byte-identical
  (`SpecAdamWConfig(fused=False)` == default). Probe-only by design: the runner's DDP/fused
  wiring + the structural `broadcast_buffers` rule is S15; `runtime_selection_executor`'s own
  DDP/optimizer folds in at S14 (its optimizer already groups and has foreach/fused/default
  variants the minimal helper does not model). Packaging: the kernel builder bundles the whole
  `src/eqvae` tree, so the new module ships automatically (all `run.py` gitignored). Import
  hygiene: dropped the orphaned `inductor_config`/`create_adamw_optimizer` from the probe,
  moved `DistributedDataParallel` to `TYPE_CHECKING` (now annotation-only). Gate 414 passed,
  basedpyright/ruff clean; adversarial review 6 lenses 0 findings; mutation-proven (a
  flat-optimizer revert fails the grouped-optimizer guard, a dropped DDP knob fails both
  DDP-knob guards). +6 tests (3 shared-module units, 3 probe-adapter).

### Phase 2 — Generator emits the compiled plan

- **S11 (DONE — this commit) Plan schema + generator payload recipe knobs.** Additive only:
  `SelectedRuntimePlan` (`training/selected_runtime.py`) gains nine OPTIONAL recipe fields with
  eager-v5 defaults — `compile_backend="eager"`, `compile_dynamic=False`, `optimize_ddp=""`,
  `compiled_autograd=False`, `reorder_compute_comm_overlap=False`, `ddp_broadcast_buffers=True`,
  `ddp_find_unused_parameters=False`, `ddp_bucket_cap_mb=None`, `fused_optimizer=False` (defaults
  verified against the runner DDP wrap `:2737` + probe `_EAGER_SPEC`). `_plan_from_payload` parses
  them from their **frozen carrier homes** (dynamo knobs → `torch_compile`; DDP/optimizer knobs →
  `runtime_policy`, beside the existing `ddp_*`), so the committed v5 plan parses byte-identically
  (every knob absent → eager default). The generator `_selected_runtime_payload`
  (`benchmarking/runtime_selection.py`) emits the knobs into those blocks, sourced from the measured
  winner row via `.get(col, eager_default)` — no matrix column exists yet, so today's rows stay
  eager (S13 adds the columns; `_bool_from_csv`/`_optional_int_from_csv` reused). S7's `_recipe_field`
  "Phase-2 carrier reconciliation" breadcrumb is resolved (homes frozen; the
  `runtime_policy`→`torch_compile`→top read order already resolves each knob from its home and stays a
  no-op on v5). **Deliberately NOT touched** (later steps, not behavior-preserving): the literal
  value-validators (`_torch_compile_errors`/`_runtime_policy_errors`/`_top_level_errors`) that would
  *accept* a compiled plan, and the observation/`expected_application`/`_application_mismatches` mirror
  (deferred past S15; see the S15 entry). Gate 418 passed (414+4), basedpyright/ruff clean; adversarial review 5 lenses → 1 low
  test-soundness finding (two safety-adjacent knobs asserted at their eager default = not
  mutation-proof) FIXED with distinguishing values, fold-delta clean. +4 tests (parser
  eager-defaults + carrier-home reads; generator eager-emission + measured-sourcing).
- **S12 (DONE — this commit) Selector: whole-step compile as a first-class candidate**,
  eligibility gated on the **relationship** (compiled AND strict settle-proof: post-settle
  `graph_break_count==0`, `recompile_count==0`, `settle_steps>=required`). Additive +
  behavior-preserving-today: adds `COMPILE_STEP="step"` (the plan token S16 keys on;
  `_selected_runtime_payload` copies `compile_scope` verbatim) admitted via
  `_STABLE_COMPILE_SCOPES={model_forward, step}` in `_compiled_row_stable`
  (`benchmarking/runtime_selection.py`), so both consumers
  (`_enforce_compiled_rows_diagnostic_only`, `_runtime_row_candidate_pass`) accept a
  settle-proven step row. The set **excludes** the diagnostic scopes
  `model_loss`/`train_step_no_optimizer` (stay diagnostic-only). INERT until S13 adds the
  `step` grid scope (config `compile_scopes` today =
  `none/model_forward/model_loss/train_step_no_optimizer`) and S14 measures it — the
  executor marks any non-`{none,model_forward}` scope `compile_scope_implementation_pending`
  (status≠pass), so no step row reaches selection yet. Gate 421 passed (418+3),
  basedpyright/ruff clean; adversarial review 6 lenses → 0 surviving findings (1 low
  S13-naming concern refuted as a later-step config item). +2 tests (settle-proven step row
  selectable, parametrized over both scopes = mutation-proof; whole-step row without
  settle-proof stays diagnostic-only).
- **S13 (DONE — this commit) Grid declares the whole-step + bigger-batch candidates;
  schema carries the recipe knobs.** GRID
  (`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`): `"step"` added to
  top-level `runtime_matrix.compile_scopes` — the ONE field the pretest reads
  (`real_data_runtime_pretest.py:1131`), so it enumerates a `step` row per seeded
  candidate, fail-closed to `compile_scope_implementation_pending` by the guard `:700`
  exactly like `model_loss`/`train_step_no_optimizer` — plus the `fp32_compile_corruption_screen`
  list (doc-sync; that top-level `stages` array + `candidate_per_device_batch_sizes` have
  NO code reader). `48` added to `candidate_per_device_batch_sizes` (declarative, like the
  un-executed bs32 precedent). `full_train_step_with_optimizer` flipped
  `out_of_scope_until_spec_update` → `in_scope_as_compile_scope_step`. The **required**
  eager `dual_t4_train_step_gate` stays `[4,8,12] × ["none"]` — untouched. **ZERO executor
  edits**: the selection executor's `_runtime_policies:406` hard-raise is never fed `step`
  (it reads `efficiency_followup.policies`, all `none`); whole-step EXECUTION stays S14.
  SCHEMA: `RUNTIME_MATRIX_COLUMNS` (`runtime_schema.py`) gains the 7 recipe-knob columns
  the S11 generator `.get`-reads (`optimize_ddp, compiled_autograd,
  reorder_compute_comm_overlap, ddp_broadcast_buffers, ddp_find_unused_parameters,
  ddp_bucket_cap_mb, fused_optimizer`), contiguous after `compile_dynamic`;
  `compile_backend` DELIBERATELY omitted (derived from `compile_scope` at `:1919`, never
  `.get`-read → would be an inert column). A new shared `EAGER_RECIPE_KNOB_COLUMNS` (eager
  v5 cell values) is spread into all 5 producers (`_runtime_rows`, pretest `_base_row`,
  executor `_base_selection_row`, `runtime_selection._runtime_row`, and the test helper)
  to close the `write_csv` `restval=''` → `_bool_from_csv('')` reload-crash trap.
  Behavior-preserving on the eager v5 path (runtime-decision fields byte-identical; a
  *regenerated* plan's provenance `selected_row_snapshot`/`runtime_matrix_sha256`
  legitimately grow with the additive columns — harmless, all consumers use fixed-subset
  `.get`). Gate 424 passed, basedpyright/ruff clean; 6-lens adversarial review → 4
  test-quality findings (0 source): the real-producer CSV round-trip guard (mutation-proven
  across all 4 producers), a de-tautologized step-scope enumeration test, and two stale
  docstrings — all fixed; fix-delta review 0 findings. +tests (real-producer round-trip,
  legacy absent-column fallback, step-scope enumeration).
- **S14** Fold the probe's feasibility sweep + winner-recipe DDP timing into
  `runtime_selection_executor` as ONE generator that emits the full linked-proof graph.
  The *executor authoring* is local + gated (behavior-preserving on the eager path);
  only the *measured dual-T4 run* is Kaggle (that run is S17). Decomposed into gated
  local sub-steps:
  - **S14a (DONE — `1dc3901`) Thread the measured recipe knobs config → plan.** The
    seven compiled fast-path recipe knobs (`optimize_ddp`, `compiled_autograd`,
    `reorder_compute_comm_overlap`, `ddp_broadcast_buffers`, `ddp_find_unused_parameters`,
    `ddp_bucket_cap_mb`, `fused_optimizer`) are added to `RowSpec` + `_RuntimePolicy`
    (eager-v5 defaults), parsed in `_runtime_policies`, threaded in `_row_spec`, and
    emitted into the selection CSV row via ONE shared `_recipe_knob_columns(row_spec)`
    producer helper — replacing the hardcoded `EAGER_RECIPE_KNOB_COLUMNS` constant spread
    in BOTH the pretest `_base_row` and the executor `_base_selection_row` — so a compiled
    winner row carries its MEASURED recipe into `_selected_runtime_payload` (S11 read the
    columns; S13 emitted eager defaults; S14a emits the measured values). `_encode_ddp_config`
    reuses `_row_spec_payload` (de-duplicated; the dual-rank child rebuilds the measured
    RowSpec, not an eager one). Behavior-preserving on the eager path: an eager RowSpec
    emits byte-identical cells, the compile-scope guards still fail-close `'step'`, and NO
    execution path changes. Gate 446 passed (ruff + basedpyright clean); adversarial review
    clean (2 fresh clean-context reviewers, 0 confirmed; all 5 threading seams
    mutation-proven, both producers guarded by the shared helper).
  - **S14b Compiled whole-step EXECUTION branch (executor dual-rank path).**
    - **Dual-T4 executor branch (DONE — `7256cd3`, local).** Opened the executor's two
      compile-scope guards (`_runtime_policies`, `_compile_ddp_model_if_requested`) for
      `'step'` and added the compiled-whole-step branch to `_run_ddp_rank_row`, mirroring
      the proven runner S16 code (`make_fastpath_step_fn` +
      `torch.compile(step_fn, dynamic=False, backend="inductor")`,
      `compiled_autograd_context`, `apply_fastpath_dynamo_config`) via the shared recipe
      helpers, consuming the S14a-threaded knobs (DDP wrap / fused / dynamo). Loop split:
      the settle loop drives the COMPILED step (so the first trace is warmed before the
      dynamo-counter reset — a forward-only settle would score the row as recompiling →
      permanently ineligible); the numerical-proof loop stays byte-identical eager (its
      mu/logvar/corruption-hash/gate lanes cannot come from the compiled
      `FastpathStepOutput`); warmup/measured route through the reduced-telemetry compiled
      step. Extracted `_build_eager_ddp_optimizer` (old inline construction verbatim) so
      the eager path is behavior-preserving. Promoted `model_requires_buffer_broadcast`
      to the shared `fastpath_recipe` module (executor + runner drive DDP
      `broadcast_buffers` from one structural rule). Two fail-closed preconditions in
      `_build_compiled_ddp_step` keep the measured recipe faithful to what the runner
      consumes: `precision_policy == amp_off_fp32` only (the compiled closure hardcodes
      fp32 / no GradScaler) and `ddp_static_graph == False` only (a step row interleaves
      an eager proof backward between compiled backwards on one DDP module, which
      static_graph forbids; the committed `model_forward` path is immune). Gate 452
      passed (ruff + basedpyright clean); adversarial review (read-only, default-refute)
      0 confirmed defects, the two latent divergence risks hardened into the guards above.
      Authored + unit-gated locally; the compiled-step throughput / zero-graph-break is a
      Kaggle observation (S17).
    - **Single-GPU pretest surface (DONE — `aabd886`, local).** Added `compile_scope=='step'`
      support to `real_data_runtime_pretest`: `COMPILE_STEP`/`_STEP_COMPILE_BACKEND`, a widened
      `_run_stage1_rows` guard (step is no longer `compile_scope_implementation_pending`), and
      single-GPU `_build_compiled_step`/`_run_compiled_step_batch` (no DDP) mirroring the dual-T4
      branch's fused optimizer + `apply_fastpath_dynamo_config` + `make_fastpath_step_fn` +
      `torch.compile(dynamic=False, backend="inductor")`. `_run_child_row` grows a
      `run_one_step(step_index, iterator)` dispatch (settle drives the compiled step so the trace
      warms before the counter reset); the eager `none`/`model_forward` path is byte-identical.
      `_model_for_compile_scope_name` returns the step model unwrapped so the paired numerical proof
      stays eager. The secondary evidence surfaces are widened (`_unique_train_step_target_rows`,
      `_compile_evidence_pass_for_row` step==model_forward parity, `implemented_compile_scopes`,
      `_compile_settle_proof` `configured_pass`), and the fail-closed ceiling is preserved: step stays
      ineligible exactly like model_forward (`settle_coverage_pass=False` is hardcoded). One fail-closed
      guard: `amp_off_fp32` only (the executor's `static_graph` guard is N/A — single-GPU has no DDP and
      the child runs no interleaved eager backward). Gate 459 (was 452), basedpyright/ruff clean; 4-lens
      read-only adversarial review → 1 LOW (`configured_pass` symmetry) fixed + 1 test-gap (compiled-step
      recipe wiring) fixed with a mutation-proven spy test; the compiled *execution* throughput /
      zero-graph-break stays a Kaggle observation (S17). +7 CPU tests.
  - **S14c (DONE — two commits, local-only, 2026-07-14; gate 472 + fix-delta adversarial
    review both green): `2927293` (C1, seam extraction) + `c59856e` (C2+C3, executor screen +
    grid + fix-delta) Feasibility sweep + grid wiring.** C1 extracted the probe's synthetic
    single-GPU-no-DDP VRAM primitives (`mem_get_info` min(free, total−peak_reserved) + 1 GB
    margin + binary-search ceiling + doubling ladder + `is_oom_error` spanning CUDA-OOM +
    cuBLAS/cuDNN alloc-failed + `NO_OOM`/`OOM` reduce sentinels) into the shared seam
    `benchmarking/vram_feasibility.py` and repointed the probe (deleting its private
    `_sweep_*`/`_binary_search_ceiling`/`_is_oom_error` copies), behavior-preserving. C2+C3: the
    executor screens each grid `compile_scope=='step'` row for VRAM feasibility in the DDP CHILD
    BEFORE the DDP build (`_screen_compiled_step_vram_feasibility`: fresh model + fused optimizer
    + `make_fastpath_step_fn(autocast_enabled=False)` + `torch.compile`, 2 synthetic-zeros steps,
    `probe_headroom_bytes` read at peak), `_all_reduce_int` SUM-reduces the per-rank infeasible
    flag so BOTH ranks take the identical skip/continue branch, and an infeasible batch writes a
    clean `oom` FAIL payload (`_vram_infeasible_rank_payload`) + `dist.barrier()` + return instead
    of a hard failure; the `oom` column propagates through `_dual_row_from_rank_payloads`→
    `_failure_row(oom=)` and `runtime_selection._runtime_row_candidate_pass` rejects any
    `oom == "true"` row. The grid gains the `'step'` + winner-recipe policy at
    `runtime_matrix.selection_benchmark_slice.efficiency_followup.policies`
    (`compile_step_ddp_optimizer_fp32_channels_last`; +bs48; `precision_policy=amp_off_fp32` +
    `compile_scope=step` + channels_last + fused + bucket_cap 50 from `_DDP_OPTIMIZER_SPEC`,
    satisfying the S14b `_build_compiled_ddp_step` guards) so the executor enumerates + measures
    the compiled winner. Fix-delta adversarial review (read-only, default-refute): Fixes A/B/C
    survived refutation — **A** gates `rank_payloads` to `()` unless the dual row is PASS +
    PASS-guards the two consumer loops (an oom row can no longer crash dual-evidence aggregation);
    **B** `_efficiency_row_enumerable` drops an AMP policy row whose batch is not in the fp32-eager
    `dual_batch_sizes` (else amp@48 with no fp32 companion blocks the write forever); **C**
    broadened `is_oom_error` for symmetric cross-rank classification — and **D**'s one LOW del-list
    omission (the trailing `FastpathStepOutput` outliving `empty_cache`) is closed (`del output`
    + `output=None` pre-init). Honor the Plan-provenance real-data requirement: the compiled rows'
    `samples_sec` (feeding `material_speedup_over_baseline`) are measured on the **same real-data
    DDP loader** as the eager baseline; the synthetic `no_dataset` path is reused ONLY for the VRAM
    feasibility verdict, never for the throughput number (a Kaggle observation, S17).

### Phase 3 — Runner consumes the recipe (plan-gated, behavior-preserving off)

- **S15 (DONE — this commit) Runner consumes the DDP recipe + fused enable via
  `fastpath_recipe.py`.** `_maybe_wrap_ddp` now routes through `wrap_fastpath_ddp`, consuming
  the plan's DDP knobs; `broadcast_buffers = plan.ddp_broadcast_buffers OR
  _model_requires_buffer_broadcast(model)` — a new structural, model-agnostic rule returning
  True iff any persistent buffer leaf is a torch running-stat name
  (`running_mean`/`running_var`/`num_batches_tracked`), so the OR can only force broadcasting
  **on** (never off): the non-eq VAE (GroupNorm + constant binomial `kernel` buffers) → False,
  a running-stat norm → True, a norm with `track_running_stats=False` → False (`named_buffers`
  drops `None` buffers). Fused: `_optimizer_config` gained `fused=plan.fused_optimizer`
  (threaded from `_settings`); **both** optimizer build sites (main run +
  `_checkpoint_resume_proof` rebuild) route through `build_fastpath_optimizer` (the now-unused
  `create_adamw_optimizer`/`DistributedDataParallel` runner imports were dropped; 3 test sites
  re-pointed). `assert_ddp_parameters_in_sync` kept. **Behavior-preserving** at the eager-v5
  plan: torch-2.12 DDP defaults (`broadcast_buffers=True`, `find_unused_parameters=False`,
  `bucket_cap_mb=None`) exactly match the eager values the old wrap omitted, and
  `fused=False → fused=None` (foreach path unchanged); the DDP branch is Kaggle-only
  (`world_size==1` early-returns unchanged). **Deferred (documented in `selected_runtime.py`):**
  the plan-applied **observation mirror** for the new knobs (a naive `observed == plan` check
  would false-flag the legitimate structural `broadcast_buffers` override) and the **dynamo
  config** (`optimize_ddp`/`compiled_autograd`/`reorder` — inert without `torch.compile`, folds
  into S16). Gate 431 passed (424+7), basedpyright/ruff clean; 6-lens adversarial review → 0
  confirmed (2 raw findings both refuted: the name-based rule is inert for every plan the runner
  consumes + safe for the planned statistics-free eq norm; the stale mirror breadcrumb is
  reconciled here). +7 tests, each mutation-proof via a `wrap_fastpath_ddp` spy (structural rule
  3-case; DDP-wrap eager behavior-preserving + distinguishing knobs + structural-override +
  single-process passthrough; fused threading ×2).
- **S16 (DONE — `3298a57`, local-only) Runner compiled whole-step path + drop_last flip
  + dynamo wire.** `_maybe_build_compiled_step` (main wiring, over the DDP-wrapped model)
  returns `None` on the eager v5 plan (`torch_compile_enabled` False / scope `"none"`) so
  the eager `_run_train_step` is byte-identical; when `plan.torch_compile_enabled AND
  compile_scope=='step'` it sets the dynamo config (`apply_fastpath_dynamo_config`, the
  S15-deferred wire), builds the SAME `make_fastpath_step_fn` closure the probe measured
  (`InlineStainCorruptor` train-only inline blake2b-free corruption + AMP forward + FP32
  loss island; `autocast_enabled=amp.enabled` matches the eager path's gating — new shared
  `make_fastpath_step_fn` kwarg, probe byte-identical at its default `True`), and
  `torch.compile(step, dynamic=False, backend=plan.compile_backend)`. `_run_compiled_train_step`
  drives it: backward/GradScaler/clip/optimizer stay eager (backward inside a shared
  `compiled_autograd_context` extracted into `fastpath_recipe.py`, probe repointed), telemetry
  reconstructed field-by-field to match the eager `_SelectedRuntimeStepResult`. The compiled
  path is exercised only via directly-constructed plans in tests (the parser still REJECTS a
  compiled plan → S17 acceptance de-pin + observation mirror + corruption-label accuracy are
  DEFERRED). **drop_last flip:** the shared `_loader` computes `drop_last` via a new
  `_safe_drop_last` guard (True for the flip; falls back to False only when a per-rank shard
  would be smaller than one batch, so a degenerate shard can never silently empty
  `_cycle_batches` → hang); applied to BOTH the `DistributedSampler` and the `DataLoader`.
  Honest provenance: the DDP sampler-policy label (`_DEFAULT_DDP_SAMPLER_POLICY` →
  `..._drop_last_true`, `_DDP_SAMPLER_POLICY_NO_DROP_LAST` added) and
  `_effective_train_epoch_samples` (floor if drop_last else ceil) both track the realized
  `_safe_drop_last` decision, not a hardcoded True. Behavior-preserving @ bs24 (train divides
  evenly, no tail; floor==ceil). Gate 441 (was 431),
  basedpyright/ruff clean. Adversarial Workflow review (6 lenses → 3 findings, all
  mutation-backed; 4 lenses [behavior-preservation, recipe-fidelity, step-correctness,
  probe-repoint] ZERO): 2 test-coverage gaps + 1 honesty-label decoupling — all fixed and
  re-mutation-proven (kl mis-map, autocast-hardcode, sampler-label-hardcode each caught by
  its guarding test). GOTCHA for next agent: the review Workflow ran in the NON-isolated
  working tree and LEFT one mutation (`ssim_weight=0.0, autocast_enabled=True`) in the source
  — audited the whole diff + reverted; use `isolation:'worktree'` OR forbid source edits in
  future review workflows (a worktree sees only committed state, so it can't review
  uncommitted work — forbid-edits is the tool for uncommitted diffs). Eager `_run_train_step`
  retained when compile off.

### Phase 4 — Activate (values flip) + full run

- **S17** Accept the compiled plan + activate the deferred observation, then run the
  generator on Kaggle. The parser-acceptance de-pin is authored **locally in gated
  sub-steps** (behavior-preserving on the committed v5 fallback), mirroring the S14 arc;
  only the row_id mint + the measured dual-T4 run are Kaggle.
  - **S17a (DONE — this commit) De-pin the recipe value-validators to a coherence
    model.** `_mixed_precision_errors`, `_torch_compile_errors`, `_runtime_policy_errors`
    (`training/selected_runtime.py`) now accept BOTH the eager v5 fallback profile
    (`amp_conservative` / `contiguous` / `compile none` / `eager` backend) AND the
    compiled winner profile (`amp_off_fp32` / `channels_last` / `compile step` /
    `inductor` backend) via allowed-set + internal-consistency checks instead of the
    eager literals: the AMP fields must agree with the declared precision policy
    (`amp_conservative` / `amp_scalar_gate_relaxed` require fp16 autocast + grad scaler;
    `amp_off_fp32` requires both off and forbids an AMP autocast dtype); torch.compile's
    enabled/scope/backend must be internally coherent (`enabled ⇒ scope ∈ {model_forward,
    step}` + `inductor`; `disabled ⇒ scope none` + `eager`) with `dynamic` always False;
    `memory_format ∈ {contiguous, channels_last}` and `ddp_gradient_as_bucket_view`
    de-pinned. Safety anchors stay pinned: the FP32 loss island is required in every
    profile, `ddp_static_graph` stays False, `zero_grad_set_to_none` stays True, and
    `_ddp_optimizer_safety_errors` is untouched; the identity comparisons tightened
    `!=`→`is` (a fail-closed improvement). Corruption strategy stays `indexed_masked`
    for both profiles (the compiled slice keeps that label; its train-fast-path label
    accuracy is S17c). Behavior-preserving: the committed v5 plan still parses with zero
    errors. **Identity (`selected_row_id` / `runtime_policy_id`) and the snapshot
    batch/precision literals are NOT touched here** — deferred to S17b + the Kaggle mint.
    Gate 497 passed (494→497), basedpyright/ruff clean; 4 read-only adversarial reviewers
    (behavior-preservation, coherence-vs-emitter, test-soundness, fix-delta) → 2 sound,
    test-soundness 2 gaps (untested `enabled=False` scope check + missing `runtime_policy`
    non-dict sentinel) + 1 coherence tightening (amp_off must reject an AMP autocast
    dtype) — all fixed and re-verified mutation-resistant; fixture-carrier fidelity
    aligned to the S11 frozen homes.
  - **S17b** Make the identity STRUCTURAL (user decision 2026-07-14: structural-now
    over Kaggle-gated re-point) + de-pin the snapshot batch/precision literals to
    cross-consistency. Decomposed into three gated local commits:
    - **S17b-1 (DONE — this commit) Parser structural identity + snapshot
      cross-consistency.** A new stdlib-only leaf `benchmarking/row_id.py` single-sources
      the selected-runtime row_id formula (`compose_row_id_base` /
      `compose_selected_row_id` / `DEFAULT_RUNTIME_POLICY_ID`); the three emitters
      (`runtime_selection`, `real_data_runtime_pretest`, `runtime_selection_executor`
      `_row_id`) delegate to it (byte-identical). In `training/selected_runtime.py`,
      `_top_level_errors`, `_snapshot_errors`, and the two `_runtime_proof_*` validators
      now check the recorded `selected_row_id` / `runtime_policy_id` against the id
      recomposed from the plan's own fields (`_composed_selected_row_id`), and the
      snapshot `bs`/`amp_conservative`/`float16`/`grad_scaler` cells cross-check the
      plan's own top-level fields; the hardware/status anchors (accelerator, machine
      shape, world size, nproc, corruption, status) stay pinned. The two top-level
      identity error ids were renamed `*_not_v5_fallback` → structural
      (`selected_runtime_selected_row_id_not_self_consistent` /
      `selected_runtime_runtime_policy_id_missing`); all snapshot/proof ids preserved.
      Behavior-preserving: the committed v5 plan recomposes to the frozen literal and
      still parses with zero errors. Gate 522 passed/1 skipped (497→522),
      basedpyright/ruff clean; 4 read-only adversarial reviewers (behavior/fail-open,
      emitter↔parser contract, snapshot edges, test-soundness) → 0 confirmed.
    - **S17b-2** (DONE — `3b72534`, 2026-07-14, local-only) Gate
      `_remote_output_gate_health_blockers` CSV-row pin → the
      `gate_health.csv` `row_id`/`candidate_row_id`/`runtime_policy_id` cells are now
      compared against the loaded plan's own identity, not `EXPECTED_SELECTED_ROW_ID`,
      via a NEW public `composed_selected_runtime_identity` that single-sources what a
      plan's identity IS (`_runtime_proof_errors` derives through it too — its empty
      `runtime_policy_id` case tightens `""`→`None` inside an already-failing payload).
      A None component fails closed. Byte-identical on the committed v5 plan, which
      parses clean and composes back to BOTH frozen constants (pinned by a test, so the
      still-published constants cannot drift from what the plan says). The gate keeps the
      `EXPECTED_*` import + `__all__` re-export (inert; tests import them from the gate,
      per the `EXPECTED_DATASET_SLUG` precedent) — zero validation logic reads them.
      **Anchor regression found by review and fixed in-step:** the literal also
      INCIDENTALLY pinned accelerator/topology on the remote path (it encodes the whole
      row shape), and the remote-output verifiers never ran the parser — so de-pinning
      identity alone let a self-declared `single_t4`/`world_size=1` plan verify CLEAN
      (reproduced, then re-blocked). Both verifiers now
      `blockers.extend(_selected_runtime_errors(payload, selected_runtime_path=...))`
      BEFORE deriving identity, restoring the `_launch_errors` anchors the de-pin's own
      rationale already assumed and enforcing strictly more than the literal did. Fixture
      lesson: the parser resolves+hash-checks `artifacts.runtime_proof` relative to the
      plan path, so a parse-clean re-mint must `copytree` the real tree and update plan +
      snapshot + proof (`selected_runtime_write_decision`/`efficiency_followup`) +
      `runtime_proof_sha256`. Gate 551 passed/1 skipped (522→551), ruff/basedpyright
      clean; 3 read-only adversarial reviewers → 4 findings adopted (anchor regression,
      a single-source divergence with `_str_or_none`, an unprotected full-output identity
      site, missing `bool`-guard coverage), 2 refuted, 1 reviewer disagreement resolved
      by direct check. Every new test mutation-proven.
    - **S17b-3 DONE** (sub-step 1 `21b697f`, 2026-07-17 + sub-step 2 `c090d16`, 2026-07-18,
      local-only) De-pin the kernel/push-side **mirrors of the parser** so a re-measured
      compiled winner builds and validates end to end. The mirrors are DIVERGENT DUPLICATES of
      `selected_runtime_plan_errors` that were never de-pinned when the parser was
      (S17a/S17b), so the fix DELEGATES to that single-source gatekeeper rather than
      hand-re-implementing its coherence model a third time. Both the full and debug RUNS
      already parse through it (`selected_runtime_runner.py`, `debug.py`), so delegation is
      zero-drift by construction and behavior-preserving on the committed v5 plan.
      - **DONE (sub-step 1):** both `run_template.py` validators
        (`_validate_baseline_selected_runtime`) now call
        `selected_runtime_plan_errors(payload, selected_runtime_path=None)` and raise on any
        error; the dead `EXPECTED_*` constants and the five drifted debug `_baseline_*`
        helpers are deleted; the kernel tests prove the wiring (accept the committed plan +
        a compiled odd-batch-47 winner, propagate the parser's rejection id, keep the
        hardware anchor). `None` skips ONLY the runtime-proof hash — the launch parse
        re-checks it and the push guard requires the proof file present, so the pre-check
        character is unchanged. NOTE: the debug run_template ALSO literal-pinned **batch
        12/24** (a third live axis the earlier four-surface map omitted — the full kernel
        de-pinned batch in S8b, the debug shell guard in S8, but the debug run_template
        never did); delegation covers it.
      - **DONE (sub-step 2 — `c090d16`, 2026-07-18):** the debug push guard in
        `scripts/kaggle_kernel.sh` (now the `PYDEBUGPAYLOAD` heredoc) runs on the venv
        interpreter (`PYTHONPATH=src "$python_bin"`, as the FULL guard already does since S8)
        and delegates to `selected_runtime_plan_errors(payload, selected_runtime_path=None)`,
        keeping its required-files / required-source-text / hardware-anchor checks; ~80 lines
        of hand-mirrored identity / recipe / snapshot / batch / AMP / dataloader / corruption
        pins were deleted (net simplification of the guard). The heredoc delimiter was renamed
        `PY`→`PYDEBUGPAYLOAD` so the test extracts the real body without drift. Tests run that
        body against a freshly built debug kernel (accept committed v5 + a bs47 amp-off
        compile-step winner, reject a self-inconsistent plan with the parser's id, keep the
        dual-T4 anchor), plus a structural test asserting BOTH push guards open on the venv
        interpreter — the load-bearing interpreter switch the body-only extraction can't see,
        a review-caught coverage gap. Gate 563/1. Three read-only default-refute reviewers:
        coverage + anchor/fail-open CLEAN; test-soundness found the interpreter gap (fixed).
      - **NO CHANGE (confirmed):** the FULL push guard (`PYFULLPAYLOAD`) pins only the batch
        relationship + schedule tokens, not identity/recipe; `build_kaggle_embedded_kernel.py`
        only bakes the (already relationship-derived) schedule tokens. The `runtime_selection`
        kernel + its shell guard + config `efficiency_followup.baseline_row_id` pin the v5
        row too, but that is the PRODUCER's baseline anchor (the reference the search runs
        AGAINST, a config-stamped constant), NOT a consumer mirror — de-pinning it would be
        the producer-audit trap (decision 0010); it stays v5. Consequence for S17-Kaggle: a
        minted compiled plan cannot naively OVERWRITE
        `runtime_selection_v5/.../selected_runtime.json` without breaking that baseline
        check — the mint must choose the committed plan path deliberately.
      **MECHANISM = compose at RUNTIME in the kernel (user decision 2026-07-14; do NOT bake
      at build time).** The kernel validators run ON KAGGLE with torch + payload on
      `sys.path` (`main()` inserts `payload_src` before validating), and function-body
      `eqvae` imports are the established kernel pattern (`noqa: PLC0415`). The S8b bake was
      NOT a precedent: it baked `FULL_TARGET_UPDATES` only because the shell GREPS that
      token; nothing greps identity/recipe. The surviving required-text anchors
      (`dual_t4_ddp`, `--nproc_per_node=2`) are hardware anchors and survive untouched.
  - **S17c (DONE — `9f6d813`, local)** Observation mirror + corruption-label: the nine S11
    recipe knobs are added to `SelectedRuntimeApplicationObservation` / `expected_application`
    / `_application_mismatches` (eight checked for exact equality; `ddp_broadcast_buffers`
    tolerates the structural UPWARD override — fires only when `plan.ddp_broadcast_buffers and
    not observed.ddp_broadcast_buffers`, and the runner feeds the EFFECTIVE
    `plan.ddp_broadcast_buffers or model_requires_buffer_broadcast(model)` in via
    `_effective_broadcast_buffers`). `local_amp_status` is plan-derived
    (`expected_local_amp_status` → `executed_amp_off_fp32` when amp is off); the compiled
    train-fast-path corruption label is accurate (`expected_corruption_strategy` →
    `compiled_fastpath_inline_stain` when `torch_compile_enabled and compile_scope=="step"`,
    else `plan.corruption_strategy` — blake2b `indexed_masked`; the `InlineStainCorruptor` was
    already applied inline in S16). The runner records the ACTUAL step:
    `compiled_step_active = compiled_step_fn is not None` drives BOTH the metric-row
    `torch_compile_enabled`/`compile_scope` AND the corruption label (the same immutable value
    that dispatches the step, so a label cannot claim compiled-while-eager); the final proof
    reads observed compile state back from the recorded rows. Byte-identical on the committed
    eager v5 plan (empirically: `_application_mismatches` → `()`, `build_plan_applied_proof` →
    local_pass). Gate 574/1; four clean-context default-refute reviewers all clean. This ends
    the `_application_mismatches` tautology that S17d's `_dataloader_errors` de-pin is gated on.
  - **S17d** Dataloader search axis (bounded, hardware-derived). The dataloader is the
    one runtime block that is still frozen at a literal while genuinely depending on the
    box — but it is NOT currently searched, and the code says otherwise in two places
    that have already caused one false de-pin. Read this before touching it:
    - **It is not measured today.** `runtime_matrix` has NO dataloader axis; the winner
      `row_id` carries no dataloader term. `runtime_selection_executor.py:1186` stamps
      all five cells from `real_data_runtime_pretest.DEFAULT_DATALOADER_*` constants
      (`0 / "" / False / False / True` — value-equivalent to the reverted pins, not
      literally identical: the `""` becomes plan `null` via `_optional_int_from_csv` at
      `runtime_selection.py:1454`), and the measuring loaders it builds at `:3349`
      and `:2005` hardcode `num_workers=DEFAULT_DATALOADER_NUM_WORKERS` and never pass
      `pin_memory` / `persistent_workers` / `prefetch_factor` at all.
    - **Two traps.** `runtime_schema.DATALOADER_MATRIX_COLUMNS` is the CSV column schema
      of a measurement RECORD, not a search grid — misreading it as a grid is what
      motivated a bogus parser de-pin on 2026-07-15 (reverted). And the `candidates`
      grid at `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json:39-45` is DEAD
      config: its only would-be reader (`dataloader_pretest.py:175`) reads a different
      key (`dataloader_pretest`) that this config does not define.
    - **Hardware ground truth (user, 2026-07-15).** `num_workers` is CPU-bound (~4 max on
      the box) and the DDP ranks SHARE one CPU budget, so the per-rank ceiling falls as
      `world_size` rises. That makes the bound DERIVED, not frozen:
      `num_workers_per_rank ≤ cpu_count // world_size`. Both worked answers are IN SCOPE
      and come from one formula — `machine_shape` is fixed at `NvidiaTeslaT4` and the
      searched axis is `accelerator_modes = ["single_visible_t4", "dual_t4_ddp"]` (config
      `:48`, `runtime_selection.py:50`), i.e. 1-vs-2 visible GPUs on the SAME box:
      `single_visible_t4` → `world_size=1` → ceiling ~4; `dual_t4_ddp` → `world_size=2` →
      ceiling ~2. Never freeze `4` — it is exactly the hardware literal this spec exists
      to eliminate and it goes stale if Kaggle re-specs the instance.
    - **P100 is OUT OF SCOPE — do not "fix" the runner to admit it.** Kaggle also offers a
      P100 (16GB, `world_size=1`), but this plan ANCHORS dual-T4: the parser pins
      `accelerator_mode: "dual_t4_ddp"` + `machine_shape: EXPECTED_MACHINE_SHAPE`
      (`= "NvidiaTeslaT4"`) at `selected_runtime.py:771-772` and `world_size ∈ {2, "2"}`
      at `:802-803`, and `tests/test_selected_runtime_runner.py:528` asserts a P100
      machine shape is REJECTED (the only other `P100` string in the repo). Admitting it
      is a SEPARATE step that must move those anchors deliberately. S17d must not touch
      them: hardware stays an anchor.
    - **Reading `cpu_count`.** Prefer torch's own helper `torch._utils.cpu_count()`
      (`torch/_utils.py:777-787` — `sched_getaffinity` first, `os.cpu_count` fallback);
      torch's DataLoader over-count warning already consumes it (`dataloader.py:607`).
      `os.cpu_count()` reports the HOST and can overshoot a cgroup-limited container —
      the exact failure the bound exists to prevent — and `sched_getaffinity` does not
      read cgroup CPU quota either, so treat the number as approximate and let the
      MEASUREMENT decide.
    - **The ceiling is an upper SEARCH bound, not a recommendation.** At `cpu_count=4,
      world_size=2` it yields 2 workers/rank = 4 worker procs + 2 main procs on 4 vCPUs —
      already oversubscribed; `(cpu_count - world_size) // world_size` is the tighter
      form. Evidence for a TIGHT budget: the historical dual-T4 notebook ran
      `num_workers: 1` (2 total workers across 2 ranks) and set `OMP_NUM_THREADS=1` /
      `MKL_NUM_THREADS=1` before `torchrun --nproc_per_node=2`
      (`docs/behavior_inventory_kaggle.md:157-158`, `:194`). One datapoint cannot
      establish the ceiling in either direction — sweep and measure, do not assume 4 is
      reachable.
    - **Work.** (a) Add the bounded axis to `runtime_matrix`; (b) make the executor
      actually sweep it and emit MEASURED cells instead of stamped constants;
      (c) reconcile the prefetch rule — `dataloader_pretest.py:253-254` raises
      "prefetch_factor is required when num_workers is positive" and an IDENTICAL second
      site exists at `:421` (fix BOTH, or the two validators still disagree); torch by
      contrast silently defaults `prefetch_factor=2` at `dataloader.py:285-286`, so pick
      one model deliberately; (d) ONLY THEN replace the parser's `_dataloader_errors`
      pins (this step owns that de-pin — it is not a separate step).
    - **Ordering.** (d) is gated on (a)+(b)+(c) AND on S17c ending the
      `_application_mismatches` tautology (`selected_runtime_runner.py:5084-5088` echoes
      `plan.*`, so the dataloader cells compare `x == x`; contrast `:5089-5092`, where
      `corruption_strategy` uses a REAL `metric_rows` observation). The replacement must
      carry the derived bound, NOT bare torch coherence: torch only rejects negatives
      (`dataloader.py:287-288`), so an unbounded model accepts `num_workers=64,
      prefetch_factor=1000` — strictly less safety than the pins, with no new capability,
      which is precisely why the 2026-07-15 attempt was reverted. Note the real boundary
      is `prefetch_factor < 1`, not `< 0`: the constructor rejects `< 0` at `:287-288`
      but `_MultiProcessingDataLoaderIter` rejects `<= 0` at `:1104-1106`, so a `0`
      constructs cleanly and detonates on the FIRST batch, deep into a paid run.
    - **Perf bar + FSQ prior art.** The proven prior art is
      `kaggle/fsq_train_reference.py` (the retained FSQ training reference; the actual
      notebook is not in-repo). The bar for S17d + S17e is **match-or-beat** it on loading
      and throughput; refactoring runtime code and re-benchmarking on Kaggle are IN SCOPE
      to hit it — do not settle for behavior-preserving-but-slower when a refactor closes a
      measured gap (a green gate proves correctness, not speed). FSQ was a FIRST attempt,
      so treat its orderings and arg combinations as CANDIDATES to measure and beat, not
      proven optima — the mmap read wrapper, the `madvise` flag, uint8-vs-pinned staging,
      and the collate strategy are all open axes. Techniques to match/adopt (each a
      hypothesis to test, not a given), with FSQ anchors:
      - *Zero-copy read (already the repo baseline; nothing to search).* Worker-local
        `mmap(ACCESS_READ)` + `MADV_SEQUENTIAL` + `torch.from_numpy`/`frombuffer` on uint8
        (`fsq_train_reference.py:160-182`); carried forward in `PatchTensorDataset` via
        `torch.frombuffer` DIRECTLY on the mmap — one view layer fewer than FSQ's
        `np.frombuffer`→`torch.from_numpy`. The wrapper (native `mmap` slice vs numpy vs
        `torch.frombuffer` vs `memoryview`) and the `madvise` flag are measurable micro-axes,
        but the read cost is the page fault, not the wrapper — and a native `mm[a:b]` slice
        COPIES, so it is likely SLOWER than a zero-copy view, not faster.
      - *Async prefetch = the sweep.* FSQ ran `num_workers=1` / `pin_memory=True` (`:87`,
        `:688`). The runner ALREADY threads `num_workers`/`prefetch_factor`/`pin_memory`/
        `persistent_workers` from the plan (`selected_runtime_runner.py:2717-2727`) +
        `non_blocking` H2D (`:4304`); only the executor stamps defaults. Sweep bounded:
        `num_workers ≤ cpu_count // world_size`, `pin_memory=True`, `prefetch_factor ∈
        {2,4}`, `persistent_workers=True`. `pin_memory` + `non_blocking` are coupled —
        `non_blocking` from pageable memory is a no-op, so flip them together.
      - *Fused H2D + layout on uint8.* FSQ does `images_uint8.to(device,
        memory_format=torch.channels_last, non_blocking=True).to(torch.float32)/127.5-1`
        (`:834`). Fusing device + `memory_format` into ONE `.to()` allocates the NHWC
        destination once (vs two allocs / passes), and reordering on uint8 BEFORE the float
        cast is 4× less reformat traffic. HONEST caveat: the CHW→NHWC reorder is a post-DMA
        GPU kernel, NOT "on the wire" (a `cudaMemcpyAsync` is a contiguous byte DMA) — the
        win is the single alloc + overlap, not a free ride on the transfer. Verify the repo
        adopts this fused-on-uint8 order (`selected_runtime_runner.py:3875-3876`/`:4304`,
        executor `:2373`); if it reorders post-cast or in two allocs, switch (same math).
      - *Transfer uint8, transform inside the compiled step.* Keep the H2D at uint8 (4×
        smaller) and FOLD the uint8→float normalize + channels_last INTO the compiled
        whole-step (S16) so `torch.compile` fuses cast+normalize+channels_last+corrupt+
        forward (corruption is already inside; OPEN: confirm normalize + channels_last are
        inside the compiled region, not an eager pre-step — if eager, folding them in is a
        measurable win). Do NOT `torch.compile` the CPU worker read. FSQ normalized OUTSIDE
        its `compiled_step` (`:834`), so the repo can go one better.
      - *Piping principle.* Prefer one op with all args over chaining — each avoided
        `.to()` / `.contiguous()` is one fewer GPU tensor + pass.
      - *Super-batch fetch + on-GPU sub-slice (candidate — measure).* Fetch a LARGER
        contiguous block from the loader (uint8 → VRAM-cheap), transfer it ONCE, then slice
        the training batch on-GPU (`chunk[i*G:(i+1)*G]`) per step. Amortizes the per-batch
        Python / collate / worker-IPC + H2D-setup overhead across fewer, bigger,
        higher-bandwidth transfers. NOT gradient accumulation — the optimizer still steps
        per G-sized slice, so the training math is unchanged; it groups the FETCH, not the
        step. The pre-shuffled bin makes a contiguous super-slice a valid shuffled batch (cf.
        FSQ's contiguous-rank slice), so this can BYPASS per-item `__getitem__` + collate (B
        Python calls + a stack per batch) via a batch-returning dataset / `BatchSampler` —
        still worker-compatible. Caveats: keep the step count at `floor(P/G)`; the tail is
        freely DROPPED (`drop_last`) so there is NO remainder to handle (losing a few
        thousand of ~300k patches is a non-issue) — and a fixed G is exactly what CUDA
        graphs need; per-rank under DDP.
      - *Objective + what to verify.* Every axis is judged by GPU utilization — throughput
        `G / step_time(G)`, i.e. minimizing GPU idle between steps (GPU-time is the binding
        constraint; this is the S17e objective, and the dataloader exists to keep the GPU
        fed). And treat the REPO's current choices as unverified too, not just FSQ's: the
        `torch.frombuffer`-vs-numpy read, the `MADV_SEQUENTIAL` flag, and the transform
        placement are all to be CONFIRMED by measurement, not assumed optimal merely because
        they already diverge from FSQ.
      - *Experimental / low-level candidates (2026-07 web research; experimental OK).*
        Ranked for the sweep (detail + sources in the `eqvae-s17d-dataloader-design` memory
        note): (1) **CUDA graphs on the compiled step** — `torch.compile(mode="reduce-overhead")`
        / `options={triton.cudagraphs:True}` kills per-kernel launch overhead, a prime win
        for a small T4 step; our `dynamic=False` + fixed G (drop_last) + no-`.item()` step
        already satisfy the constraints (RNG must be cudagraph-safe; the H2D partitions out
        naturally); NOT currently set in the recipe. (2) **`cudnn.benchmark=True`** — FSQ used
        it (`fsq_train_reference.py:32`); fixed shapes → safe conv autotune; verify the repo
        sets it. (3) **Side-stream double-buffer prefetcher** — the ONLY way to overlap H2D
        with COMPUTE (default-stream `non_blocking` overlaps CPU, not the compute stream);
        needs a non-default stream + pinned source. (4) **`cudaHostRegister` the mmap
        super-slice** → DMA straight from the page cache, skipping the copy-to-pinned staging.
        (5) **GPUDirect Storage** (`torch.cuda.gds`, torch 2.7+ / `kvikio`) — direct NVMe→GPU,
        bypass CPU; our raw `.bin` is GDS-friendly, but nvidia-fs is likely UNAVAILABLE on
        Kaggle T4 (check). (6) `torch.from_file` / `UntypedStorage.from_file` torch-native
        mmap vs numpy. Also RE-VERIFY `amp_off_fp32` vs `amp_fp16` on T4 tensor cores — the
        FSQ reason for disabling autocast was its fp32 QUANTIZER, which the normal VAE lacks.
      - *Make them OPTIONS + hard rules (user 2026-07-19).* All of the above are configurable
        KNOBS (runtime_matrix axes / recipe flags) SEARCHED for the epoch-time winner, not
        hardcoded. Objective = MINIMIZE TIME PER EPOCH, tail-agnostic. Compile time is a
        NON-COST (~30h run) → default the step to `torch.compile(mode="max-autotune")`
        (exhaustive Triton/GEMM autotuning + cudagraphs, SUBSUMING the reduce-overhead
        candidate); compile-mode {`default`, `reduce-overhead`, `max-autotune`,
        `max-autotune-no-cudagraphs`} is itself a searched knob. HARD RULE: **minimize
        graph breaks** — compile the step with `fullgraph=True` to fail-fast on SPURIOUS breaks
        (a break kills the cudagraph benefit; the branchless forwards already target 0 —
        check on a SINGLE-GPU replica). DDPOptimizer's INTENTIONAL bucket-boundary breaks are
        a separate overlap trade-off, reconciled by `compiled_autograd` +
        `optimize_ddp="python_reducer"`. Determinism is NOT required — BLANKET (user: "we
        dont care for determinism") → set/expose ALL speed-over-reproducibility flags:
        `cudnn.benchmark=True`, no `use_deterministic_algorithms`, and the FASTEST RNG for
        corruption/noise (drop the per-sample blake2b seeding — a hash cost that only bought
        reproducibility; InlineStain's fast Philox can default for BOTH paths, retiring
        blake2b, a deliberate change that touches S17c's label logic). Small drift OK. Check
        the resume-prefix metric validators (non-deterministic corruption won't bit-match),
        and note cudagraphs still need cudagraph-safe (functionalized) RNG. A single global
        `set_seed` STAYS fine (FSQ seeds for identical DDP-rank init while keeping
        `benchmark=True` — seeding ≠ determinism). Precision is fp16-FIRST — keep fp32 only
        where
        numerically required, and RE-AUDIT the FSQ fp32 islands (likely over-conservative;
        the normal VAE has no fp32 quantizer, so amp_fp16 may reclaim T4 tensor-core
        throughput vs the current `amp_off_fp32` winner).
      - *Dual-GPU grad-sync trap + compile cost (user 2026-07-19; 2026-07 web).* max-autotune
        tunes SINGLE-GPU kernels; a whole fwd+bwd graph PREVENTS grad/compute overlap
        (all_reduce fires from autograd hooks only AFTER the full backward). Overlap needs the
        DDP knobs (already in the recipe): `optimize_ddp="ddp_optimizer"` (DDPOptimizer breaks
        at bucket boundaries — inserts breaks, TRADES against the fullgraph rule, couples
        breaks↔`ddp_bucket_cap_mb`) OR `compiled_autograd=True` + `optimize_ddp=False` —
        EXACTLY what FSQ used (`fsq_train_reference.py:668-669`): compiled_autograd traces
        backward+comm so DDPOptimizer's breaking is unneeded → BREAK-FREE overlap that ALIGNS
        with the minimize-graph-breaks rule. A **bf16 gradient-compression comm hook** also
        halves all_reduce bytes (fits "drift OK"). cudagraphs
        (max-autotune) + DDP have KNOWN conflicts (pytorch#113809) → may need
        `max-autotune-no-cudagraphs`; DDPOptimizer fails on custom `autograd.Function`
        (pytorch#166305). MUST measure on the REAL dual-T4 — single-GPU autotune misleads on
        the sync cost. The repo benchmark picked `ddp_optimizer_whole_step` (1.42x,
        compiled_autograd=False, INSERTS breaks) — the OPPOSITE of FSQ's break-free combo,
        likely not in that sweep → RE-MEASURE both (FSQ's aligns with the hard rule). Also
        accumulate metrics IN-GPU to avoid per-step `.item()` host-sync (FSQ
        `VarianceAccumulator`, `:609`) — a per-step `.item()` stalls the pipeline AND breaks
        cudagraphs; GENERALIZE to a family of GPU-resident accumulators for ALL telemetry
        (sync once per flush boundary, never per step). Compile time is free during the SWEEP
        too (total < one full run) → measure
        every config at the real mode, do NOT rank-cheap-run-expensive (mode + precision
        change `step_time`).
  - **S17e** (producer follow-up — spec-only, added 2026-07-17) Make the executor's batch
    axis an EXACT throughput search instead of the coarse pool
    `candidate_per_device_batch_sizes = [4,8,12,32,48]`. Objective = minimize epoch
    wall-clock `floor(P / G) × step_time(G)`, i.e. MAXIMIZE global throughput
    `G / step_time(G)`, subject to VRAM feasibility. Caveat (user, 2026-07-17): a larger
    batch with a slower per-STEP time can still win the EPOCH (more samples per step →
    fewer steps), and "largest VRAM-feasible" (the existing
    `binary_search_feasible_ceiling`) is NOT necessarily fastest — throughput can plateau
    or dip below the VRAM ceiling — so the search must MEASURE throughput across batches
    and pick the max, per-(model × hardware × recipe). `drop_last=True` means the winning
    batch need not divide P (e.g. 47 is fine). S17b-3 is the prerequisite: the consumers
    now accept any exact batch the search returns.
  - **S17f** (LOCAL, added 2026-07-19) Audit + CORRECT the current runtime code against the
    speed-first / FSQ-floor intent — the runtime was written before that intent was sharp, so
    it probably embeds slower / reproducible choices. Refactoring and re-benchmarking are
    authorized (user). CORRECT the unambiguous violations LOCALLY + gate them BEFORE the
    Kaggle search (so the search measures corrected code); leave genuinely measurement-
    dependent choices as Kaggle search AXES. Bar = match-or-beat `kaggle/fsq_train_reference.py`
    (detail in the `eqvae-s17d-dataloader-design` memory). Audit → if wrong, fix:
    - **Compile mode.** The step compiles with NO mode → add `mode="max-autotune"` (compile
      time is a non-cost for a ~30h run) as the default, and make compile-mode a searched knob
      {`default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`}. Enforce
      `fullgraph=True` on a single-GPU replica to kill SPURIOUS graph breaks.
    - **RNG — retire blake2b, sub-commit 2 DONE (`5dde097`, local, 2026-07-20).** The runner's eager
      training + validation corruption moved off the blake2b `StainCorruptor` onto the vectorized
      `InlineStainCorruptor`; corruption is now a FIXED speed-first property, not a plan-selected axis
      (user confirmed drop-the-axis — inline is certainly faster than a Python-loop-with-blake2b, no
      re-measure). `InlineStainCorruptor.forward` gained an optional keyword-only `generator`; None (the
      compiled fast path) emits the exact seedless `torch.rand`/`torch.randn_like` ops via `_rand`/
      `_randn_like`, so the measured compiled recipe stays byte-identical. TRAINING (eager
      `_run_train_step`): a per-rank generator seeded `corruption_seed+rank`, checkpointed as
      `train_corruption` and DDP-resume re-based (mirrors the eps `train_data` generator) so a resume
      CONTINUES the corruption stream, never replaying it. VALIDATION (`_validation_view_row`): a dedicated
      generator re-seeded to a fixed constant each boundary sweep → identical corruption every half-epoch =
      a stable best-checkpoint ruler, no checkpoint state. `expected_corruption_strategy` →
      `eager_inline_stain`/`compiled_fastpath_inline_stain`; the parser launch pin relaxed to a fail-closed
      structural check + the snapshot corruption pin dropped (v5 plan still parses; provenance echoes keep
      the plan's declared value). Distribution unchanged (same alpha/beta/noise ranges + where-select). The
      handoff's "make the cross-checks `isclose`" item was a NON-ISSUE (the only bit-exact corruption checks
      are blake2b-vs-blake2b in the benchmark, untouched). Gate 586/1; four clean-context default-refute
      reviewers → 0 confirmed. CAVEAT (rule 29 — NOT retired, verified separate): the fixed-25 EVAL
      determinism (`artifacts/fixed25_equivariance.py`: `posterior_mu_deterministic`, seeded eval eps) AND
      the validation `_zero_eps` latent knob (stays 0) are intentional reproducible PAPER artifacts; the
      resume-prefix validators (`_validate_full_resume_*_prefix`) are existence-only (no metric bit-compare),
      so free-running training corruption is resume-safe. Accepted rule-30 consequence: a pre-S17f checkpoint
      (only the `train_data` key) can't resume across the new `{train_data, train_corruption}` key set — no
      such checkpoint exists (the compiled full run is the first paper run), so no migration machinery.
      **FOLLOW-UPS (SCOPE CORRECTION rule 29 — blake2b was NOT dead after this runner-only change):** the
      benchmark selection corruption proof (`runtime_selection_executor` / `real_data_runtime_pretest`
      branchless-vs-`indexed_masked` hashes), `debug.py`, `kaggle_smoke.py`, and `stain_corruptor_qa.py`
      still use blake2b `corrupt_normalized_batch`. Move those off blake2b (or retire the now-vestigial
      selection corruption proof), THEN delete the dead subsystem (`corruption/stain.py`:
      `derive_corruption_seed`, `sample_corruption_parameters`, `StainCorruptor._apply_indexed_masked`, the
      `indexed_masked` strategy) + its tests.
    - **cuDNN — DONE (`a7feae4`, local, 2026-07-20).** `cudnn.benchmark=True`/`deterministic=False`
      is now a FIXED speed-first flag wherever convolutions run on GPU (not a searched axis): new
      shared `fastpath_recipe.apply_cudnn_flags`; runner `_apply_cuda_runtime_flags(device)` (CUDA-
      only) wired into `_distributed_context`; executor `_RuntimePolicy` + pretest `RowSpec`
      `cudnn_benchmark` field+parse defaults false→true (executor applies via the existing
      `_apply_backend_policy`; the pretest child applies from its row_spec → records what it runs;
      the config declares no cuDNN key). `deterministic` stays False. Behavior-preserving on v5:
      cuDNN is not a plan/parser/gate/row_id field, so the runner ignores stale cuDNN cells and no
      validator/provenance hash changes; the local_cpu synthetic row + the v5 reference row stay
      benchmark=false (honest — neither runs cuDNN). Gate 581/1, basedpyright clean; 6 mutation-proof
      tests; 3 clean-context adversarial reviewers (correctness/premise, speed-first/honesty, test-
      soundness) → 0 defects. The pretest apply's invocation is CUDA-only (Kaggle-exercised),
      consistent with the repo's pretest coverage boundary. `use_deterministic_algorithms(True)` is
      confirmed absent everywhere. Still keep one global `set_seed` for DDP-rank-identical init
      (seeding ≠ determinism) — that lives with the RNG item, not this one.
    - **DDP grad overlap.** Search FSQ's break-free `compiled_autograd=True`+`optimize_ddp=False`
      vs `ddp_optimizer` (measure BOTH on dual-T4; respect the S7 no-`ddp_optimizer`+
      `compiled_autograd` constraint); cudagraphs+DDP may force `max-autotune-no-cudagraphs`.
    - **Full validation — DONE (`a6c6271`, local, 2026-07-20).** The full run now sweeps the
      WHOLE validation dataset every half-epoch (matching FSQ), not a capped 20-batch leading
      slice — the old cap made best-checkpoint selection use a fixed, non-representative slice
      of the unshuffled/slide-grouped validation bin (a distinct issue the FSQ comparison
      surfaced; the cap was an agent decision, not the user's). `validation_batches_per_view=0`
      = full sweep; a positive value caps (debug/dry-run). New `_validation_batches(loader, cap)`
      helper; `_validation_view_row` emits the ACTUAL swept `batch_count`. Gate de-pinned:
      summary sentinel 0 + per-row `batch_count > 0` (data-dependent count, no validation-size
      ground-truth to pin); the full-run anchor requires 0 so a real run can't silently cap.
      Corrected the non-current capped-probe decision text in Spec 0009 (AC#5 + Open Questions)
      + this spec (rule 15). DDP: each rank sweeps its shard once, negligible per-rank tail, the
      cross-rank sample-weighted selection metric is unchanged. Gate 583/1; +2 mutation-proof
      tests (helper + CPU end-to-end full-sweep); 3 clean-context reviewers + fix-delta clean.
    - **Metrics — part 1 DONE (`623f128`, local, 2026-07-20).** The three hot-step
      per-parameter host-sync loops in the runner (`_global_grad_norm`,
      `_nonfinite_gradient_count`, `_parameter_update_norm`) now reduce on-device to ONE
      `.item()` each (was O(N_params)/step), and the update-norm drops the per-parameter
      `.cpu()` D2H copies; both the eager and compiled train-step paths benefit (shared
      helpers, byte-identical call sites). Value-preserving: the non-finite count is exact,
      the two norms drift only within fp tolerance (only finiteness is gate-checked), and the
      per-step CSV/gate contract is untouched (`_clone_trainable_parameters` unchanged). Gate
      595/1; five clean-context default-refute reviewers (value-preservation,
      correctness/edge, consumer/gate-contract, test-soundness, fix-delta) → 0 confirmed; 9
      mutation-proof tests. The perf intent (single host sync) is Kaggle-observed, not
      CPU-unit-testable. **PART 2 — 3 commits V → T → C; V and T DONE, C remaining.** Intent:
      FSQ's `VarianceAccumulator` (`fsq_train_reference.py:609`)
      AGGREGATES across steps — zero per-step sync, but it emits interval-aggregated rows and has
      NO per-step training row. That maps onto VALIDATION (already one mean-row per view) but NOT
      onto TRAINING: the remote gate `_remote_full_train_step_blockers` audits per-step rows
      (`row_count == target_updates × world_size`; every step `1..target` present on every rank),
      so we CANNOT aggregate the training path — we buffer-but-keep per-step. Three commits,
      order V → T → C (each a gated commit + clean-context adversarial review):
        - **Commit V — validation aggregate + std — DONE (local, 2026-07-21; gate 602/1,
          ruff+basedpyright clean; 6-lens clean-context adversarial review green — the one
          confirmed finding, a coverage gap where no test caught deletion of the shared per-view
          `zero_()` reset, was fixed + mutation-proven; +5 tests).** `_validation_view_row` accumulates the 6
          loss tensors' `sum` AND `sum_sq` into an on-device fp64 buffer (reused across the 2
          views: allocate once in `_run_scheduled_validation`, `.zero_()` per view); `beta`
          hoisted out of the batch loop (host float, loop-invariant, no sync); ONE `.tolist()`
          per view → mean and std on host. `6 × n_batches` syncs/view → 1. Emits NEW `*_std`
          columns (ADDITIVE — the gate's required-column check is a subset test; update
          `_VALIDATION_METRIC_COLUMNS` + confirm the remote verifier tolerates extra columns) so
          each half-epoch validation point gets an error bar for plotting. Means value-preserving;
          `_boundary_selection_metric` (reads the row's mean l1 + sample_count) untouched.
        - **Commit T — training per-step buffer — DONE (`0ddc857`, local, 2026-07-22; gate
          604/1, ruff+basedpyright clean; 4 clean-context default-refute reviewers
          [value-preservation / ordering-timing-flush / sync-graph-memory /
          consumer-contract-test-soundness] → 0 confirmed).** The step metric fields on
          `_SelectedRuntimeStepResult`/`_ReconstructionOutputStats` are 0-dim device tensors, and
          the RUNNER's `_global_grad_norm`/`_nonfinite_gradient_count`/`_parameter_update_norm`/
          `_reconstruction_output_stats` RETURN tensors (the `step.py` /
          `runtime_selection_executor.py` duplicates untouched). A persistent
          `_TrainStepMetricBuffer` `[save_every_steps, 14]` **fp32** buffer index-writes each
          step's 14 device scalars (6 losses + grad_norm + param_update_norm + 5 recon stats +
          nonfinite_count) IN PLACE, column by column (no `torch.stack` temporary), with NO host
          sync — detached, so it retains no autograd graph — and
          `_metric_row` reads them from one bulk `.tolist()` at the half-epoch flush (an auto-flush
          on overshoot handles AMP skips; a tail flush drains a partial final window). ~14 metric
          syncs/step → ~0 (amp-off) or 1 (fp16 GradScaler inf-check floor, = FSQ); touches BOTH the
          eager and compiled step paths. The 2 eps stats STAY host floats — computed on the CPU eps
          tensor, they are never a device sync, so buffering them would add GPU work for zero sync
          benefit (rule 29), which is why the buffer is 14 wide, not the 16 first sketched. DTYPE
          (user call): the training buffer is **fp32** because it only STORES — one write, one read
          per row, so no accumulation error can build up, and fp32 already carries the full
          precision of these fp32-origin metrics; **fp64 is reserved for the Commit V validation
          accumulator**, which sums `sum`+`sum_sq` across batches and so does risk cancellation.
          fp32 keeps the 6 losses / `x_hat_*` / `frac_*` columns byte-identical (fp32-origin) and
          rounds only the three fp64-reduced norms (fp-tolerant telemetry, rule 30);
          `nonfinite_count` stays exact (fp32 is exact below 2^24 vs the model's 3.96M params, and
          every consumer tests only zero-vs-non-zero). KEEPS every per-step row (gate contract +
          whole-file atomic CSV write unchanged); CSV schema byte-identical; amp-agnostic. +2
          mutation-proof buffer tests. The training curve stays dense per-step → error bands are a
          plot-time rolling std (nothing extra to store).
        - **Commit C — one CSV per half-epoch.** Shard `train_steps.csv` into a file per
          half-epoch, killing the O(n²) whole-file atomic rewrite (`_write_csv_atomic` rewrites
          the whole accumulated list every boundary) + the unbounded single file. Each shard is
          named by its boundary so shards never overwrite each other — a zero-padded
          `train_steps_<optimizer_step:06d>.csv.gz` keyed on the half-epoch boundary step, which
          also glob-sorts chronologically. Reuse the repo's EXISTING boundary-keyed naming
          convention (`checkpoints/step_*.pt`, `artifacts/fixed25/boundary_<step:06d>/`), which is
          the internal analog of the FSQ reference's per-boundary `ep_<ep>.<fraction>` keying
          (`fsq_train_reference.py:999`). FSQ instead APPENDS to one CSV (`append_to_csv`,
          `mode='a'`, `:604`); we splinter into atomic gzipped shards rather than append because
          the repo's crash-safe atomic-replace write cannot append in place. Write the shards
          gzip-compressed (per-step float rows compress ~10×) against the Kaggle ~20 GB
          `/kaggle/working` output cap (see FU-046).
          CONTRACT change: `_remote_full_train_step_blockers` + the remote verifier read a SINGLE
          file today → they must glob (`train_steps_*.csv.gz`), decompress, + concatenate the shards
          in boundary order. Its OWN commit — the only part-2 commit that changes
          a gate READER contract (the train-step file layout). (Commit V's `*_std` addition also
          extends the validation column schema, but it is backward-compatible: additive columns pass
          the gate's subset-based required-column check, so V updates `_VALIDATION_METRIC_COLUMNS` +
          needs a verifier tolerance check without breaking the contract. Commit T keeps every
          per-step training row, so it changes NOTHING in the gate contract. `_norm`/`a_grad_norm`
          in `_gate_health_rows` is a boundary-only sync via `_write_interval_artifact_flush`, not
          the hot path — out of scope.)
    - **Transforms — DONE (`2ce6a4c`, local, 2026-07-20).** `make_fastpath_step_fn` now takes
      `x_uint8` and folds the uint8->float normalize into the compiled graph
      (cast+normalize+corrupt+forward+loss fused); every compiled caller (runner, probe,
      executor + VRAM screen, pretest) transfers/synthesizes uint8 (channels_last fused into
      the H2D `.to()` in the runner's `_to_device`), and the pretest + executor dataloader-H2D
      proofs time the uint8 transfer. Eager paths keep CPU-normalizing (blake2b-coupled → the
      RNG item). The CPU worker read stays eager, by design (do not compile it). Gate 575/1,
      basedpyright clean; 3 clean-context adversarial reviewers (fold-correctness / caller-
      completeness / scope) clean after 2 fixes. The speed win is Kaggle-measured (local CPU-only).
    - **Precision — its OWN gated step, NOT folded into Metrics.** The `amp_off_fp32` default
      across the T4 grid (`runtime_selection.py` candidates; `model_loss_train_step.py:40`
      `_REQUIRED_PRECISION_POLICY`) was an agent's unilateral "to be safe" choice, NOT the
      user's — the user always optimizes for max speed, so fp16 (T4/Turing tensor cores ~2x;
      bf16 unsupported on sm_75) is the likely-intended precision. Re-measure `amp_off_fp32` vs
      `amp_fp16` on T4 and move to fp16, gated FIRST on a rule-29 check that amp-off was not set
      for a real NaN/divergence reason (the one legitimate reason to keep fp32). On fp16 the
      GradScaler inf-check is the one unavoidable per-step sync (the FSQ floor).
    - **`drop_last` UNIT-FLIP — DONE (`3b9aa42`, local, 2026-07-20).** Flipped the
      stale selection-benchmark PROJECTION-RECORD to match the real loader (`drop_last=True` since
      S16) + `floor(P/G)` schedule (S5b): now `drop_last=true`, `effective_samples_per_epoch =
      steps_per_epoch * global_batch_size`, `remainder_samples` + the partial-batch path DROPPED,
      `projection_basis` → `floor_steps_times_global_batch_drop_last_true`. Sites: CODE
      `runtime_selection._global_projection_payload` + `synthetic_timing.py`
      (`SYNTHETIC_TIMING_MATRIX_COLUMNS` dropped `remainder_samples` + `_base_row`) +
      `real_data_runtime_pretest.py` `missing_coverage` (dropped `final_partial_batch_path`,
      mirroring `kaggle_cli_workflow.md`); CONFIG `non_eq_vae_kaggle_runtime_benchmark.json`
      (`drop_last:true`, `must_exercise` dropped `final_partial_batch_path`); DOCS Spec 0001
      (`:1297`/`:1334`/projection block), decision 0008, `kaggle_cli_workflow.md`; TESTS
      `test_synthetic_timing.py` effective `299968`/`299904` (gb 64/128 don't divide P → floored,
      forced). Untouched by design: runner `_DDP_SAMPLER_POLICY_NO_DROP_LAST` degenerate-shard
      fallback + pretest `partial_batch_observed` mechanics probe. Guard-health steady 3 FAIL/21.
      The tail is dropped and does not matter (rule 30).
  - **S17-Kaggle** [Kaggle] Run the S14 generator on dual-T4 → new compiled
    `selected_runtime.json` (winner row_id
    `dual_t4_ddp__bs48__amp_off_fp32__compile_step__indexed_masked__policy_…`). With the
    identity made STRUCTURAL in S17b (parser done in S17b-1; gate/`run_template` in
    S17b-2/3), NO anchor re-point is needed — the de-pinned consumers accept the minted
    id because it is self-consistent with the plan's own fields. Only the emitted plan
    itself is new.
- **S18** Docs: de-pin Spec 0009 schedule passages to formulas (bs24 as a worked
  example only); add decision record `docs/decisions/0010-...` framing the mechanism
  as reusable across architectures; update CURRENT.md / specs README / open_follow_ups.
  Keep lean (net trim CURRENT.md).
- **S19** [Kaggle, ~30h + ~30min staging] First paper-promotable compiled full run;
  the relational gate certifies it against its own goal-derived schedule + LR
  relationship, failing closed on any truncation/gap.

## Decisions (resolved; not open forks)

- **Plan provenance** → (a) folded honest generator (only honest + reusable).
- **Batch source** → re-measured by the generator each run (never a consumed 48).
- **LR rule** → sqrt default (AdamW), per-model reference in the model config; user
  re-tunes `reference_lr`/`rule` per model. Baseline unchanged at scale 1.
- **`drop_last`** → **`True` for BOTH train and validation — DONE (S16 `3298a57`; the stale
  projection-record then unit-flipped in S17f `3b9aa42`).** Tails do not matter (drop_last drops
  a negligible per-rank remainder). Validation sweeps the FULL dataset every half-epoch (S17f
  `a6c6271` — `validation_batches_per_view = 0`, matching the FSQ
  reference), and its `sum(l1*n)/sum(n)` reduction stays correct across shards, so
  best-checkpoint selection uses the whole validation set (not a fixed leading slice).
  The sealed masked-WSI TEST set is a separate future evaluator.
  `updates_per_epoch = floor(P/G)` single-sourced (matches the train flip).
- **Odd updates_per_epoch** → floor half + the shared boundary generator driving BOTH
  producers and consumers, with the terminal validated (user decision). Do NOT
  constrain the selector to even-updates batches (that would distort the eq model's
  true optimal batch). Drop the even-guard.
- **Model seam** → one dict registry keyed on `model.kind`, opaque per-kind kwargs (not
  a `norm_groups` universal signature); no larger abstraction.

## Acceptance Criteria

- Phases 1 + 3: `./scripts/python_quality.sh` green after each step; every derived
  schedule/LR value at the current v5 plan (global 24) is byte-identical to today's
  (12500 / 125000 / 6250 / 6250 / beta 12500 / lr 0.0005); odd-batch coverage is proven
  by (a) a pure-function boundary-generator unit test (`half=1562, target=31250`,
  terminal force-included) and (b) a tiny odd full config CPU run
  (`updates_per_epoch=5, target=5`, terminal off-grid) — NOT a literal global-96 DDP
  run (infeasible on CPU).
- Phase 2/4: the folded generator emits a `selected_runtime.json` whose
  `full_run_eligible=true` is backed by real dual-T4 measured rows + linked proofs and
  a material speedup over the honest eager small-batch baseline.
- S19: strict `--verify-full-output` certifies the compiled run against its own
  derived schedule + LR relationship with zero launch blockers.

## Tests And Verification Commands

- `./scripts/python_quality.sh` (ruff ALL + basedpyright strict + pytest) per step.
- New odd-updates tests in `tests/test_selected_runtime_full_run.py`: a pure-function
  boundary-generator unit test + a tiny odd full config CPU run (`updates_per_epoch=5`,
  `target=5`, terminal off-grid), single-process. The literal global-96 world_size=2
  DDP case is a Kaggle-only observation (CPU collapses to world_size=1).
- Kaggle: `--verify-full-output` gate on the downloaded run (S19).
- Local (CPU) authors + unit-tests compile/DDP/fused paths with compile off / fused
  off / contiguous; zero-graph-break / grad-sync / speedup are Kaggle-only observations.

## Known Risks / Adversarial Checks (verified against the code — 2026-07-06)

- **drop_last↔schedule (HIGH):** the real harm is **recompilation of the
  `dynamic=False` compiled step** (a short tail batch reaches the static-shape step),
  NOT a DDP tail desync — `DistributedSampler(drop_last=False)` pads ranks equally, so
  the collective stays symmetric. Fix = flip the train loader to `drop_last=True`
  (currently `False`) + single-source `floor`; the four `ceil(P/G)` sites
  (`runtime_selection.py:1866/1043`, `synthetic_timing.py:1361/1915`) all become the
  floor helper.
- **Frozen-schedule homes — de-pin status (all Phase-1 homes now converted):**
  `kaggle_kernel.sh` (MF1) was the one BROKEN today — the FULL guard's config loop
  (`:2136-2148`), **not** run_template — **de-pinned in S8**. Plan parser `_launch_errors`
  in **S7**; runner `_validate_full_run_settings` in **S5**; gate `REMOTE_FULL_*` +
  generator `ceil` in **S5b/S6**; run_template `_validate_baseline_selected_runtime` + the
  `FULL_*` constants (now build-derived) in **S8b**. A re-measured non-24 plan is now
  accepted end-to-end — the builder, the shell guard, and the run.py validator all derive
  the schedule from it.
- **Gate self-anchor (MF2):** without an independent `updates==floor(P/G)` assert
  against the *immutable single-sourced* P (not the plan number, not the config
  override), a de-pinned gate could certify a self-consistent dataset-coverage shrink.
- **Odd-batch boundary (MF3) — the spec's own fix can misfire:** the terminal is NOT
  dropped (final.pt saves it; modulo producers + range consumers already agree on the
  last grid step). Applying `| {target}` to consumers ONLY (producers stay modulo)
  false-rejects a valid run. The shared generator must drive BOTH producers and
  consumers so the terminal is written+validated+expected symmetrically.
- **DDPOptimizer safety (ADD, not keep):** the parser has NO such invariant today
  (only `ddp_static_graph`). Add: reject `optimize_ddp=='ddp_optimizer'` with
  **`compiled_autograd`** / `static_graph` / `find_unused_parameters` True. The
  memory-caught **silent** all_reduce drop (each rank an independent replica) is the
  `compiled_autograd=True` case — the winner `_DDP_OPTIMIZER_SPEC` pairs DDPOptimizer
  with `compiled_autograd=False` — so omitting it would leave the exact silent failure
  the guard exists to catch uncovered (an early draft that guarded only
  `static_graph`/`find_unused` guarded the *loud* dynamo-#93672 / bucket-split cases and
  missed the silent one; corrected here). `broadcast_buffers=False` is safe for the
  baseline (6 constant binomial-kernel buffers, verified) but the structural
  constant-buffer check is mandatory so a future eq model with a mutated buffer flips it
  to True instead of silently desyncing.
- **Honesty:** `full_run_eligible` must be earned by real-data linked proofs (the probe
  is non-promotable; its synthetic timing must never feed the speedup — S14); report
  the gain as a combined recipe+batch improvement over the eager small-batch baseline
  (there is no same-batch eager baseline at bs48 — it OOMs on T4, a Kaggle-only
  observation to confirm on-device), not as "1.4× compile".

## Open Questions

None blocking Phase 1. `REAL_TRAIN_PATCH_COUNT` is currently 4-way duplicated +
config-overridable; S5/S6 must pick ONE canonical immutable source and re-point the
rest (the gate anchor imports it). The exact activation batch is a measured output of
S14/S17, not a decision.

## Related Files

- `GOAL.md`, `docs/repo_goal_and_requirements.md`, `docs/equivariant_vae_transition_plan.md`
- Spec 0009 (superseded schedule parts), Spec 0010 (fixed-25), `docs/decisions/README.md`
- Memory: `specs-encode-goals-not-frozen-numbers`, `eqvae-compiled-ddp-optimize-ddp`,
  `eqvae-fast-path-speed-priority`
- Design workflow output (detailed steps + critique):
  `.../54c70b5b-.../tasks/wvhy5430o.output` (session-local; distilled here)
