# Spec 0011: Reusable goal-derived runtime mechanism + compiled fast-path

Status: draft active — Phase 1 (S1–S10) + Phase 2 (S11–S13) DONE committed through `8e14650`; Phase 3 S15 DONE (this commit, local-only)
Implementation readiness: Phase 3 S16 local, implementation-ready; Kaggle phases S14/S17/S19 gated (user-driven)
Owner/workstream: selected-runtime speed + reusability
Last updated: 2026-07-10 (S15 done; NEXT = S16). The per-step `(DONE — …)` tags in the body are the state of record.

## Purpose

Make the training runtime a **reusable, re-tunable mechanism** rather than a set
of frozen numbers, and use it to promote the compiled fast-path + bigger batch
into the first paper-promotable full run.

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
static_graph=False)` · fused AdamW · drop the blake2b semantic-seed corruption on
the **train** fast path (keep it on validation/deterministic paths). The non-eq VAE
has 6 persistent buffers (`FixedBinomialLowpassDownsample2x.kernel`), so
`broadcast_buffers` matters — and its correct value is **model-specific** (must be
driven by a structural buffer check, not a hardcoded flag).

## Architecture / Workflow Contract

### Goal-derived relationships (the invariants every validator/gate enforces)

Let `P = REAL_TRAIN_PATCH_COUNT = 300000`. **P is NOT single-sourced today** — it
is defined four times under three names (`synthetic_timing.py:85`,
`runtime_selection.py:73` `REAL_TRAIN_PATCH_COUNT_DEFAULT`, and
`fixed32_selector_readiness.py:37` + `real_data_runtime_pretest.py:58`
`EXPECTED_REAL_TRAIN_PATCH_COUNT`) and is *config-overridable* in the generator
(`runtime_selection.py:352`, `data.real_train_patch_count or DEFAULT`). S5/S6 must
pick ONE immutable canonical constant, re-point the other three as imports, and the
gate anchor (MF2) must read THAT — never the plan's number, never
`data.real_train_patch_count`. `G = global_batch = per_device_batch * world_size`,
`E = epochs` (policy anchor).

- `global_batch == per_device_batch * world_size`
- `updates_per_epoch == floor(P / G)` — **floor**, matching the post-flip
  train-loader `drop_last=True`. The train loader is `drop_last=False` today; **S16
  (Phase 3) flips it** (see the drop_last decision). `ceil(P/G)` currently lives at
  four sites (`runtime_selection.py:1866` [plan emitter] + `:1043`,
  `synthetic_timing.py:1361` + `:1915`) — all become the single-sourced floor helper;
  the tiny-selector `ceil` at `runner:627` is unrelated and stays.
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
  constant is not rewritten exactly once; (2) the builder runs under **bare `python3` (no
  torch)**, and importing `eqvae.benchmarking.schedule` pulls torch via the package
  `__init__` — so it **loads the two stdlib-only leaves (`schedule.py`, `roots.py`) by file
  path** (`_load_leaf_attr`, `sys.modules`-registered) to reuse the single source without
  torch (never re-inlines `P`). `_validate_baseline_selected_runtime` de-pinned to a
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
- **S14** [Kaggle] Fold the probe's single-GPU feasibility sweep (physical-free-VRAM
  gate, 1GB margin) + winner-recipe DDP timing into `runtime_selection_executor` as
  ONE generator that emits the full linked-proof graph. Honor the Plan-provenance
  real-data requirement above: the compiled candidate rows' `samples_sec` (the numbers
  feeding `material_speedup_over_baseline`) are measured on the **same real-data DDP
  loader** as the eager baseline; the probe's synthetic `no_dataset` path is reused
  ONLY for the VRAM feasibility sweep, never for the throughput number.

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
- **S16** Compiled step (`torch.compile(step, dynamic=False)` when `plan.compile_scope=='step'`)
  with train-only inline corruption (drop blake2b on train; keep it on
  validation/deterministic). **Flip the shared `_loader` to `drop_last=True`** for BOTH
  train and validation (currently `False` — a real change, not a keep; user decision:
  1 dropped step/epoch is meaningless, and validation is a capped relative probe whose
  `sum(l1*n)/sum(n)` reduction stays correct with even shards). The real payoff is
  avoiding a short tail batch that would recompile the `dynamic=False` step — the DDP
  path stays symmetric either way (the sampler pads ranks equally, so no desync).
  Correctness comes from `updates_per_epoch = floor(P/G)` single-sourced (Phase 1), so
  the static batch dim needs no padding. Eager `_run_train_step` retained when compile
  is off. (100% coverage is the future test evaluator's job, not these loops.)

### Phase 4 — Activate (values flip) + full run

- **S17** [Kaggle] Run the S14 generator on dual-T4 → new `selected_runtime.json`
  (winner = compiled bigger-batch row; new row_id). Re-point every row_id anchor.
  The reusable generator produces the artifact; de-pinned consumers accept it.
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
- **`drop_last`** → **set `True` for BOTH train and validation** (currently `False`
  everywhere — a real flip, not a keep). Flip the shared `_loader` default. Safe:
  validation is a capped ~20-batch relative probe, and its `sum(l1*n)/sum(n)` reduction
  stays correct with even shards. Behavior-preserving at bs24 (no train tail;
  validation's first 20 cycled batches are all full). 100% coverage is reserved for the
  **future sealed test evaluator** (doesn't exist yet), not the training loops.
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
