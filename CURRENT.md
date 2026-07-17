# Current Repository Status

Last updated: 2026-07-16

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE using a repo-owned,
   compile-compatible implementation, with `escnn` as a reference.

Latest handoff, 2026-07-09/10 — reusable goal-derived runtime + compiled fast-path
(**Spec 0011**, `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`):
the training runtime becomes a REUSABLE per-(model×hardware) SEARCH-then-run
mechanism (an efficiency search emits `selected_runtime.json`; the full-run config
carries no batch/LR of its own; validators/gates check relationships, not literals),
so the future equivariant model re-runs it unchanged. Committed this workstream
(Phase 1, local-only, each step gate + adversarial-review green): **B1 `119c8fd`**
un-froze the config + runner schedule derivation; then **S1–S10** built the mechanism
incrementally — S1 model-registry/latent seam, S2–S4 LR primitives + CUDA-guarded
fused AdamW, S5–S6 runner + gate schedule/LR relationship validators, and **S7
`bcc3ec0`** de-pinned the shared plan parser `_launch_errors`
(`src/eqvae/training/selected_runtime.py`) from the batch/schedule literals to
goal-derived relationships (`global == per_device × world_size`;
`updates_per_epoch == floor(REAL_TRAIN_PATCH_COUNT / global)` via single-sourced
`training_steps_per_epoch`) plus a DDPOptimizer safety invariant (reject
`optimize_ddp='ddp_optimizer'` paired with `compiled_autograd`/`static_graph`/
`find_unused_parameters` True; no-op on the v5 plan). Behavior-preserving at batch 24;
latest gate 414 passed, basedpyright clean; the granular per-step tracker lives in the
`eqvae-reusable-runtime-mechanism-plan` memory. Known B1 side effect: the build-time packaging validator
`scripts/kaggle_kernel.sh:2136-2162` (+ in-kernel `run_template.py`
`_validate_selected_runtime`/`_validate_full_config`, `tests/test_kaggle_embedded_kernel.py`)
still pins the removed literals — invisible to pytest (shell), only bites at
full-kernel build; de-pinned in Spec 0011 Phase 1. Design = 17 steps / 4 phases +
adversarial critique (5 must-fixes) folded into Spec 0011. Decisions: provenance =
one folded honest generator (probe sweep merged into `runtime_selection_executor`);
LR = sqrt, per-model reference in the model config; `drop_last=True` kept →
`updates_per_epoch = floor(P/G)` single-sourced; model seam = one dict registry on
`model.kind`. **S8 `208c92f` de-pinned the `scripts/kaggle_kernel.sh` build-time push
guards (the FULL-guard config loop B1 actually broke → un-broke preflight+push; a scout
correction: `run_template.py` was NOT broken), and S8b `0ae0188` de-pinned
`run_template.py` itself — the kernel builder now regex-derives the baked
`FULL_TARGET_UPDATES`/`FULL_HALF_EPOCH_INTERVAL` from `floor(P/global)` (importing
`schedule.py`/`roots.py` normally — the file-path `_load_leaf_attr` loader and the
"torch-less build" premise are GONE as of 2026-07-15; see the spec 0001 package/import
policy), and the run.py validator checks the batch/updates relationships — so a
re-measured non-24 full batch is now accepted end-to-end.** Then **S9 `35e40be`
routed all eight full-run schedule boundaries (2 runner producers + 4 runner consumers +
2 gate consumers) through one shared `boundary_steps` generator, so the terminal update
is a genuine boundary on BOTH the producer and consumer sides (never consumer-only, which
would false-reject a resume) — byte-identical @ batch 24, off-grid-verified by a tiny
`epochs=1/updates_per_epoch=5` CPU run; adversarial review closed a vacuous-consumer-test
finding with three mutation-proven off-grid tests.** Then **S10 `23d6854`
extracted the shared fast-path recipe module `training/fastpath_recipe.py` from the probe —
one source for the grouped/CUDA-gated fused optimizer, the DDP wrap, and the dynamo config,
taking plain scalar knobs so the runner adopts it bit-identically in S15; the probe's fused
optimizer, formerly a bespoke FLAT ungrouped AdamW that decayed norms/gates, now routes the
grouped path (matching the runner).** Phase 1 (S1–S10) is DONE. **Phase 2 opens with S11
(this commit): the `SelectedRuntimePlan` schema + the honest generator payload
(`_selected_runtime_payload`) now carry the nine compiled fast-path recipe knobs
(`optimize_ddp`, `compiled_autograd`, `reorder_compute_comm_overlap`, compile `backend`/`dynamic`,
`ddp_broadcast_buffers`/`ddp_find_unused_parameters`/`ddp_bucket_cap_mb`, `fused_optimizer`) as
OPTIONAL fields with eager-v5 defaults, parsed from frozen carrier homes (dynamo → `torch_compile`,
DDP/optimizer → `runtime_policy`) and sourced from the measured winner row via
`.get(col, eager_default)`. Additive only: the committed v5 plan parses byte-identically (all nine
default to the eager recipe); the literal value-validators (compiled-plan acceptance) and the
observation/`_application_mismatches` mirror (S15) are untouched; S7's `_recipe_field`
carrier-reconciliation breadcrumb is resolved. Gate 418 passed, basedpyright/ruff clean; adversarial
review 5 lenses → 1 low test-soundness finding (two safety-adjacent knobs asserted at their eager
default) fixed with distinguishing values, fold-delta clean.** Then **S12 (this commit)
admits whole-step compile as a first-class selection candidate: `COMPILE_STEP="step"` (the
plan token S16 keys on, copied verbatim by `_selected_runtime_payload`) is admitted via
`_STABLE_COMPILE_SCOPES={model_forward, step}` in `_compiled_row_stable`, so both consumers
(`_enforce_compiled_rows_diagnostic_only`, `_runtime_row_candidate_pass`) accept a
settle-proven step row on the same fail-closed settle relationship; the diagnostic scopes
`model_loss`/`train_step_no_optimizer` stay excluded. INERT until S13 adds the `step` grid
scope and S14 measures it (executor marks non-`{none,model_forward}` scopes
`compile_scope_implementation_pending`). Gate 421 passed (418+3), basedpyright/ruff clean;
6-lens adversarial review → 0 surviving findings; +2 tests.** Then **S13 (`8e14650`, 2026-07-10)**
declares the whole-step + bigger-batch candidates in the benchmark grid and makes
`RUNTIME_MATRIX_COLUMNS` carry the 7 S11 recipe knobs: `"step"` added to top-level
`runtime_matrix.compile_scopes` (the one pretest-read scope field → a per-seeded `step`
row, fail-closed to `compile_scope_implementation_pending` by the guard, like the other
diagnostic scopes) + the fp32-screen list (doc-sync); `48` added to the declarative
`candidate_per_device_batch_sizes` pool (bs32 precedent, no code reader); the
`full_train_step_with_optimizer` marker flipped to `in_scope_as_compile_scope_step`; the
required eager `dual_t4_train_step_gate` untouched at `[4,8,12] × ["none"]`; ZERO executor
edits (whole-step EXECUTION is S14 — the `_runtime_policies:406` hard-raise is never fed
`step`). A shared `EAGER_RECIPE_KNOB_COLUMNS` (eager v5 values; `compile_backend` omitted
as derived-not-`.get`-read) spread into all 5 producers closes the `write_csv`
`restval=''` → `_bool_from_csv('')` reload-crash trap. Behavior-preserving on the eager
path (runtime-decision fields byte-identical; a regenerated plan's provenance
snapshot/`runtime_matrix_sha256` grow with the additive columns, harmless). Gate 424,
basedpyright/ruff clean; 6-lens review → 4 test-quality findings (0 source) fixed +
fix-delta review 0 findings; the real-producer CSV round-trip guard is mutation-proven
across all 4 producers. Phase 1 + S11–S13 stay local + behavior-preserving @ batch 24.
**S15 (`357ada6`, local-only):** the runner now
CONSUMES the plan's DDP recipe + fused-optimizer knobs via `training/fastpath_recipe.py`.
`_maybe_wrap_ddp` routes through `wrap_fastpath_ddp` with `broadcast_buffers =
plan.ddp_broadcast_buffers OR _model_requires_buffer_broadcast(model)` — a new structural,
name-based rule (True iff a persistent buffer leaf is a torch running-stat name, so the OR
only forces broadcasting ON; the non-eq VAE's GroupNorm + constant binomial `kernel` buffers
→ False). Both optimizer build sites (main + `_checkpoint_resume_proof`) route through
`build_fastpath_optimizer` with `fused = plan.fused_optimizer` (threaded via
`_optimizer_config`, CUDA-gated inside `create_adamw_optimizer`); the now-unused
`create_adamw_optimizer`/`DistributedDataParallel` runner imports were dropped and 3 test
sites re-pointed. `assert_ddp_parameters_in_sync` kept. Behavior-preserving at the eager-v5
plan (torch-2.12 DDP defaults `broadcast_buffers=True`/`find_unused=False`/`bucket_cap_mb=None`
match the old omitted-kwarg wrap; `fused=False → None`). Deferred (noted in
`selected_runtime.py`): the plan-applied observation mirror (the structural override would
false-flag a naive `observed == plan` check) and the dynamo config (inert without compile →
S16). Gate 431 passed (424+7), basedpyright/ruff clean; 6-lens adversarial review → 0
confirmed. **S16 (`3298a57`, local-only):** the runner now has the
plan-gated compiled whole-step path. `_maybe_build_compiled_step` (main wiring, over the
DDP-wrapped model) returns `None` on the eager v5 plan (`torch_compile_enabled` False, scope
`"none"`) so the eager `_run_train_step` is byte-identical; when
`plan.torch_compile_enabled AND compile_scope=='step'` it applies the dynamo config
(`apply_fastpath_dynamo_config`, the S15-deferred wire), builds the SAME `make_fastpath_step_fn`
closure the probe measured (`InlineStainCorruptor` train-only inline blake2b-free corruption +
AMP forward + FP32 loss island; `autocast_enabled=amp.enabled` matches the eager path's autocast
gating — a new shared-module kwarg, probe byte-identical at its default `True`), and
`torch.compile(step, dynamic=False, backend=plan.compile_backend)`. `_run_compiled_train_step`
drives it (backward/GradScaler/clip/optimizer stay eager, backward inside a shared
`compiled_autograd_context` extracted into `fastpath_recipe.py` with the probe repointed;
telemetry reconstructed field-by-field to match the eager step). The compiled path is exercised
only via directly-constructed plans in tests — the parser still REJECTS a compiled plan, so
acceptance + the observation mirror + corruption-strategy label accuracy stay DEFERRED to S17.
The shared `_loader` is flipped to `drop_last=True` via a new `_safe_drop_last` guard (falls back
to `False` only when a per-rank shard would be smaller than one batch, so a degenerate shard can
never silently empty `_cycle_batches` → hang); the DDP sampler-policy label
(`..._drop_last_true`/`..._drop_last_false`) and `_effective_train_epoch_samples` (floor/ceil)
both track the realized `_safe_drop_last` decision, not a hardcoded value. Behavior-preserving at
bs24 (train divides evenly; validation leading batches full; floor==ceil). Gate 441 passed (was
431), basedpyright/ruff clean. 6-lens adversarial Workflow review → 3 mutation-backed findings
(2 test-coverage gaps + 1 honesty-label decoupling; the other 4 lenses — behavior-preservation,
recipe-fidelity, step-correctness, probe-repoint — ZERO); all 3 fixed and re-mutation-proven.
Provenance note: that review Workflow ran in the NON-isolated working tree and left one source
mutation, which the gate did not catch (it was the missing-coverage finding); the full diff was
audited and the mutation reverted. Phase 3 is now COMPLETE. **S14a (`1dc3901`, local-only):** S14 (fold the
probe into the executor + run a compiled `step` row) is being authored locally in gated sub-steps
ahead of the paid Kaggle run. S14a threads the seven measured compiled fast-path recipe knobs
(`optimize_ddp`, `compiled_autograd`, `reorder_compute_comm_overlap`, `ddp_broadcast_buffers`,
`ddp_find_unused_parameters`, `ddp_bucket_cap_mb`, `fused_optimizer`) from the efficiency-search
policy config through `_RuntimePolicy` → `RowSpec` → the selection CSV row via ONE shared
`_recipe_knob_columns(row_spec)` producer helper (replacing the hardcoded
`EAGER_RECIPE_KNOB_COLUMNS` spread in BOTH the pretest `_base_row` and the executor
`_base_selection_row`), so a compiled winner row carries its MEASURED recipe into
`_selected_runtime_payload` (S11 read the columns, S13 emitted eager defaults, S14a emits the
measured values); `_encode_ddp_config` reuses `_row_spec_payload` (de-duplicated → the dual-rank
child rebuilds the measured RowSpec). Behavior-preserving on the eager path: the compile-scope
guards still fail-close `'step'` and no execution path changes. Gate 446 passed, basedpyright/ruff
clean; adversarial review clean (2 fresh clean-context reviewers, 0 confirmed; 5 threading seams
mutation-proven). **S14b dual-T4 executor branch (`7256cd3`, local-only):** opened the executor's
two compile-scope guards (`_runtime_policies`, `_compile_ddp_model_if_requested`) for `'step'` and
added the compiled-whole-step branch to `_run_ddp_rank_row`, mirroring the runner S16 code
(`make_fastpath_step_fn` + `torch.compile(step_fn, dynamic=False, backend="inductor")`,
`compiled_autograd_context`, `apply_fastpath_dynamo_config`) via the shared recipe helpers and
consuming the S14a-threaded knobs. Loop split: the settle loop drives the COMPILED step (warms the
first trace before the dynamo-counter reset — a forward-only settle would score the row as
recompiling → permanently ineligible); the numerical-proof loop stays byte-identical eager (its
mu/logvar/corruption-hash/gate lanes cannot come from the compiled `FastpathStepOutput`);
warmup/measured route through the reduced-telemetry compiled step. Extracted
`_build_eager_ddp_optimizer` (eager path byte-preserving) and promoted
`model_requires_buffer_broadcast` to the shared `fastpath_recipe` module. Two fail-closed
preconditions in `_build_compiled_ddp_step` keep the measured recipe faithful to what the runner
consumes: `precision_policy == amp_off_fp32` only (compiled closure hardcodes fp32 / no GradScaler)
and `ddp_static_graph == False` only (a step row interleaves an eager proof backward between
compiled backwards on one DDP module; the committed `model_forward` path is immune). Gate 452
passed, basedpyright/ruff clean; adversarial review (read-only, default-refute) → 0 confirmed, the
two latent divergence risks hardened into the guards above. **S14b single-GPU pretest surface
(`aabd886`, local-only):** added `compile_scope=='step'` support to `real_data_runtime_pretest`,
mirroring the dual-T4 branch on the single-GPU pre-screen path — `COMPILE_STEP`/
`_STEP_COMPILE_BACKEND`, a widened `_run_stage1_rows` guard (step no longer
`compile_scope_implementation_pending`), and single-GPU `_build_compiled_step`/
`_run_compiled_step_batch` (no DDP; fused optimizer + `apply_fastpath_dynamo_config` +
`make_fastpath_step_fn(autocast_enabled=False)` + `torch.compile(dynamic=False, backend="inductor")`).
`_run_child_row` grows a `run_one_step(step_index, iterator)` dispatch (iterator passed, not captured,
so `finally: del iterator` stays; settle drives the compiled step to warm the trace before the
counter reset); the eager `none`/`model_forward` path is byte-identical. `_model_for_compile_scope_name`
returns step unwrapped so the paired numerical proof (`_one_strategy_train_step_evidence`, a separate
model) stays eager. Secondary surfaces widened (`_unique_train_step_target_rows`,
`_compile_evidence_pass_for_row` step==model_forward parity, `implemented_compile_scopes`,
`_compile_settle_proof` `configured_pass`); fail-closed ceiling preserved — step stays ineligible
exactly like model_forward (`settle_coverage_pass=False` hardcoded). One guard: `amp_off_fp32` only
(the executor's `static_graph` guard is N/A single-GPU — no DDP, no interleaved eager backward). Gate
459 (was 452), basedpyright/ruff clean; 4-lens read-only adversarial review (behavior-preservation 0,
compiled-fidelity 0, completeness 1 LOW fixed [`configured_pass` symmetry], test-soundness gap fixed
with a mutation-proven recipe-wiring spy test); the compiled EXECUTION throughput / zero-graph-break is
a Kaggle observation (S17). +7 CPU tests. **S14c DONE (two commits, local-only, 2026-07-14;
gate 472 + fix-delta adversarial review both green): `2927293` (C1) + `c59856e` (C2+C3).**
C1 extracts the probe's OOM-safe VRAM primitives into a shared seam
`src/eqvae/benchmarking/vram_feasibility.py` (`feasibility_ladder`/
`binary_search_feasible_ceiling`/`probe_headroom_bytes` = `min(free, total − peak_reserved)`
via `cuda.mem_get_info`/`headroom_below_margin` @ `VRAM_MARGIN_MB=1024`/`is_oom_error` spanning
CUDA-OOM + cuBLAS/cuDNN alloc-failed/`NO_OOM`+`OOM` reduce sentinels) and repoints the probe
(deletes its private copies), behavior-preserving. C2+C3: the executor screens each grid
`compile_scope=='step'` row for single-GPU no-DDP VRAM feasibility in the DDP CHILD BEFORE the
DDP build (`_screen_compiled_step_vram_feasibility`: fresh model + fused optimizer +
`make_fastpath_step_fn(autocast_enabled=False)` + `torch.compile`, 2 synthetic-zeros steps,
headroom read at peak — synthetic ONLY for the verdict, NEVER throughput), `_all_reduce_int`
SUM-reduces the per-rank infeasible flag so BOTH ranks take the identical skip/continue branch,
and an infeasible batch writes a clean `oom` FAIL payload (`_vram_infeasible_rank_payload`) +
`dist.barrier()` + return instead of a hard failure; the `oom` column propagates through
`_dual_row_from_rank_payloads`→`_failure_row(oom=)` and `runtime_selection._runtime_row_candidate_pass`
rejects any `oom == "true"` row. The grid `efficiency_followup` gains bs48 + a 2nd policy
`compile_step_ddp_optimizer_fp32_channels_last` (precision_policy=amp_off_fp32, compile_scope=step,
channels_last, fused, bucket_cap 50 from `_DDP_OPTIMIZER_SPEC`) satisfying the S14b
`_build_compiled_ddp_step` guards. Fix-delta adversarial review (read-only, default-refute):
Fixes A/B/C SURVIVED refutation with documented attempts — **Fix A** gates `rank_payloads` to `()`
unless the dual row is PASS + PASS-guards the two consumer loops (an oom row can no longer crash
dual-evidence aggregation); **Fix B** `_efficiency_row_enumerable` drops an AMP policy row whose
batch is not in the fp32-eager `dual_batch_sizes` (else amp@48 with no fp32 companion blocks the
write forever; the amp_off_fp32 step winner passes); **Fix C** broadened `is_oom_error` to
cuBLAS/cuDNN alloc failures for symmetric cross-rank classification — and **Fix D**'s one LOW
del-list omission (the trailing `FastpathStepOutput` outliving `empty_cache`) is closed (added to
the del + `output=None` pre-init). Behavior-preserving on the eager path (no step row selected
yet). The compiled-step EXECUTION / feasibility verdict is a Kaggle observation. **S17a DONE
(`3ed4ba6`, 2026-07-14, local-only):** the S17 parser-acceptance de-pin is S14-sized, so decomposed
into local gated sub-steps ahead of the paid run; S17a de-pinned the recipe value-validators
(`_mixed_precision_errors` / `_torch_compile_errors` / `_runtime_policy_errors` in
`training/selected_runtime.py`) from the eager v5 literals to a coherence model that ACCEPTS both the
eager fallback and the amp_off_fp32 compiled-step winner (allowed-set + internal-consistency), while
keeping every safety anchor pinned (fp32-loss island required in each profile, `ddp_static_graph`
False, `zero_grad_set_to_none` True, `_ddp_optimizer_safety_errors` untouched; the `!=`→`is` switch
tightened it). Behavior-preserving: the committed v5 plan still parses with zero errors. Identity
(`selected_row_id` / `runtime_policy_id`) + the snapshot batch/precision literals are deferred to
S17b + the Kaggle mint. Gate 497/1, basedpyright/ruff clean; 4 read-only adversarial reviewers
(behavior-preservation / coherence-vs-emitter / test-soundness / fix-delta) all clean.
**S17b-1 DONE (this commit, 2026-07-14, local-only; user chose STRUCTURAL-now over Kaggle-gated
re-point):** the parser identity is now STRUCTURAL, not literal-pinned. A new stdlib-only leaf
`benchmarking/row_id.py` single-sources the selected-runtime row_id formula (`compose_row_id_base` /
`compose_selected_row_id` / `DEFAULT_RUNTIME_POLICY_ID`); the three emitters (`runtime_selection`,
`real_data_runtime_pretest`, `runtime_selection_executor` `_row_id`) delegate to it (byte-identical).
In `training/selected_runtime.py`, `_top_level_errors` / `_snapshot_errors` / the two
`_runtime_proof_*` validators check the recorded `selected_row_id` / `runtime_policy_id` against the
id recomposed from the plan's own fields (`_composed_selected_row_id`), and the snapshot
batch/precision cells cross-check the plan's own top-level fields; the hardware/status anchors
(accelerator, machine shape, world size, nproc, corruption, status) stay pinned. The two top-level
identity error ids were renamed `*_not_v5_fallback` → structural
(`selected_runtime_selected_row_id_not_self_consistent` /
`selected_runtime_runtime_policy_id_missing`); all snapshot/proof ids preserved. Behavior-preserving:
the committed v5 plan recomposes to the frozen literal and parses with zero errors. Gate 522/1
(497→522), basedpyright/ruff clean; 4 read-only adversarial reviewers (behavior/fail-open,
emitter↔parser contract, snapshot edges, test-soundness) → 0 confirmed.
**S17b-2 DONE (`3b72534` + docs `d600465`, 2026-07-14, local-only):** the gate's downloaded
`gate_health.csv` `row_id`/`candidate_row_id`/`runtime_policy_id` cells are compared against the
loaded plan's own identity via a NEW public `composed_selected_runtime_identity` (single-sources
plan identity; `_runtime_proof_errors` derives through it too). A None component fails closed;
byte-identical on the committed v5 plan, which composes back to both frozen constants (pinned by a
test). **Review caught an anchor regression, fixed in-step:** the literal also INCIDENTALLY pinned
accelerator/topology there (it encodes the whole row shape) and the remote-output verifiers never
ran the parser — so a self-declared `single_t4`/`world_size=1` plan verified CLEAN (reproduced,
then re-blocked). Both verifiers now run `_selected_runtime_errors` BEFORE deriving identity,
restoring the `_launch_errors` anchors the de-pin's own rationale already assumed. Gate 551/1
(522→551); 3 read-only reviewers → 4 adopted / 2 refuted; every new test mutation-proven.
**NEXT LOCAL: S17b-3** — de-pin the kernel/push-side MIRRORS of the parser. **Scope re-measured
2026-07-14: this is ~S17b-2-sized, not a constant swap, and spans FOUR surfaces** (accurate map in
the spec body under S17b-3): both `run_template.py` files pin identity **and** recipe, and the real
"validates at push time" gate is the Python heredoc at `scripts/kaggle_kernel.sh:1854-1904`, which
was never listed; the `runtime_selection` copy previously named here **does not exist**. BOTH axes
are required — identity alone leaves the `amp_off_fp32` winner failing `kaggle_kernel.sh:1867` and
`run_template:315`, so the step would read as done while the run stayed blocked. **MECHANISM =
compose at RUNTIME in the kernel, NOT bake at build time (user decision). The old "build is
torch-less" argument was doubly wrong: a build/run-time conflation (validators run on Kaggle with
torch on `sys.path`) AND a self-inflicted premise (the builder now imports `eqvae` normally as of
2026-07-15). Full evidence in the spec's S17b-3 body; do not re-derive it.** Then **S17c**
(observation mirror + corruption-label accuracy), then **S17d** (bounded dataloader search axis —
NEW 2026-07-15, spec-only so far). Because identity is now STRUCTURAL, the
Kaggle-minted compiled id needs NO anchor re-point — the de-pinned consumers accept it as
self-consistent.
**A `_dataloader_errors` de-pin was attempted 2026-07-15 and REVERTED (adversarial review): the
dataloader is NOT a searched axis. Do not retry it — read S17d's traps in the spec first.**
**THEN KAGGLE-ONLY (user-driven, FRESH window, needs exact remote cmd +
`KAGGLE_PUSH_CONFIRMED=1`):** the S17 generator run → new compiled `selected_runtime.json`; S19
(~30h + ~30 min staging, push-then-monitor, NOT a held session); plus the queued LR-finder (~200–300
lines, needs real dual-T4).
**PACKAGING PREREQUISITE DONE (2026-07-16) — 5 commits, pushed to `origin/main` @ `d663acd`.**
`a09027c` gave `pyproject.toml` a hatchling `[build-system]` so
`uv sync` **editable-installs** `eqvae`: `import eqvae` now resolves from `.venv/bin/python`
for any invocation, so `PYTHONPATH=src` is redundant (harmless, not swept). This removed the
root cause of four workarounds — including the builder's file-path `_load_leaf_attr` loader
(deleted; it now `import eqvae` normally) and the self-inflicted "the BUILD must stay
torch-less" premise: the builder runs on the venv interpreter via `kaggle_kernel.sh`'s
`build_kernel_py()`, which fails closed if the venv lacks torch+eqvae. `43a2d24` recorded the
backend in Spec 0001 (which had MANDATED a spec update when a backend was added) and Spec 0002.
`64b781a` **moved `src/nn` → `reference/nn`** so the editable `.pth` cannot leak it into import
space — it is RETAINED reference material (a user decision; Spec 0002 rationale), NOT dead code
(its symptoms — nothing imports it, the gates exclude it, the payload skips it — are the
deliberate policy). `e2bbedb` added `docs/decisions/0010` (verify the premise before changing a
pin), indexed and cited by AGENTS.md rule 29. Gate green throughout (552 passed/1 skipped, ruff
and basedpyright clean); each change adversarially reviewed.
**KAGGLE SMOKE VERIFIED 2026-07-16 — the one thing local verification could not prove:** rebuilt
`run.py` through the new `build_kernel_py()` + builder path (`BUILD_EXIT=0`; plain `import eqvae`,
no `_load_leaf_attr`), then `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_REMOTE_CONFIRMED=1
./scripts/kaggle_kernel.sh push kaggle/kernels/setup_smoke --wait` → `eqvae-setup-smoke` v2
settled `COMPLETE`. Downloaded `runs/kaggle/setup_smoke/benchmark/kaggle_setup_smoke.json` has
`status = smoke_pass`, `git_commit = d663acd`, `git_dirty = false`, `dataset_slug = ""`,
`device = cpu`, 3 train steps + 1 validation batch on synthetic data — proving the payload
unzips and `import eqvae` works where NO venv exists, on the exact packaging commit, no ~30-min
staging.
**PUSH-TO-ORIGIN STANCE UPDATED 2026-07-16:** the user granted STANDING permission to push to
`origin` ON REQUEST (previously forbidden). Still never a bare `git push` (could resolve to the
`overleaf` remote) — use `git push origin <branch>`; Kaggle/Overleaf writes still need their own
confirm flags.
**Next sequence:** origin push ✓ and `setup_smoke` Kaggle proof ✓ both DONE 2026-07-16 →
**doc-hygiene IN PROGRESS 2026-07-16: clearing the `docs/open_follow_ups.md` backlog before the
run (the D-05..D-17 trims target `kaggle_kernel.sh` GUARD FILES — extract anchors from the script;
the DO NOT DROP list is incomplete; see memory `eqvae-doc-trim-backlog-clear`). Items 1–4 DONE
local-only (`1fdfbbd`/`d62b951`/`1ab48dc`/`abaeac4`; see git log for scope). Then the
`docs/kaggle_cli_workflow.md` campaign DONE local-only in 3 gated commits (each: guard-health
BEFORE/AFTER diff + `agent_preflight` + two read-only adversarial reviews + lesson-preservation
audit): `0a0e171` (D-13/D-14 banner + Current-State post-mortems, and RESTORED the two
`c02b538`-dropped guard tokens `runtime_selection_kernel_ready`/`selected_runtime_debug_gate_contract_ready`
in the two Local-first push blocks → S17 runtime-selection/debug push guards un-blocked,
guard-health 5→3 FAIL), `f0eb081` (D-11/D-12/D-17 selection + pretest sagas → pointers to the
dual-T4 selection-gate contract; closed FU-010 by deleting the contradictory v3 row-id block, v5
identity kept), `ad36684` (D-16 auth narration → durable OAuth mechanism; D-15 timing logs → cadence
rules). Retired FU-010 + FU-032 + D-11..D-17 from the backlog. Guard-health now 3 FAIL, ALL 3
by-design/other-doc (spec0001:657 + README:663/680), zero for `kaggle_cli_workflow.md`. NEXT doc-trim
= spec 0003 `D-08` → README `D-10` (anchor-DENSE) → spec 0001 `D-05` (surgical)/`D-06`/`D-07`; per-trim
procedure + remaining order live in that memory** → dependency upgrade as its OWN gated step
(`uv lock --upgrade`; ahead of S17b-3 because Kaggle rides near-latest torch and drift bites the
compiled fast-path) → S17b-3.

> Pre-Spec-0011 status (runtime-selection v5, Spec 0006–0009) and older Kaggle-run
> provenance were trimmed from here on 2026-07-16 (FU-011). Git history and
> `docs/specs/0006`–`0009` hold that detail; the live handoff is the top of this file.

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
