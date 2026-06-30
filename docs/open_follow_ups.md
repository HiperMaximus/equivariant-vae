# Open Follow-Ups

A living, **open-only** backlog of things found during review/development that are
worth coming back to. Two parts:

1. **Follow-ups** — correctness / robustness / hygiene items to fix, each with
   specific file:line so the next agent knows exactly what and where.
2. **Make-It-Current Drop Plan** — historical/narration content to delete or
   compress so the repo reflects only its current state (git holds history).

Rules to keep this file from rotting (the problem it exists to fix):
- **No "Resolved" section.** When an item is done, delete its entry — git holds it.
- Keep entries terse and file-specific. Promote anything that grows into real work
  into a `docs/specs/` spec.
- Severity: `high` (correctness/run-invalidating) · `med` · `low` · `process`
  (docs/hygiene) · `env` (tooling).

## DO NOT DROP (load-bearing — restated for the editing agent)

- `runs/kaggle/selected_runtime_debug_v5/` — proven debug/tiny gate, referenced by
  the full-run config and `scripts/kaggle_kernel.sh` guard.
- `runtime_selection v5` row id (`dual_t4_ddp__bs12__amp_conservative...`) — locked
  selected runtime.
- Guard literals grepped by `scripts/kaggle_kernel.sh`: `kaggle_smoke_ready`,
  `kaggle_setup_smoke_ready` (`:533/673`), `Implementation readiness:` (`:515`),
  and each spec `.md` filename literal in `docs/specs/README.md`.
- Benchmark code/CLIs/kernels/tests (`synthetic_timing`, `real_data_runtime_pretest`,
  `runtime_selection*`) — FINISHED but still imported/wired; retire only as one
  coordinated change (see FU-031), not in a doc pass.
- `src/nn/` (user decision), all `configs/spec0001/*.json`,
  `docs/repo_goal_and_requirements.md`, `docs/issue_image_inventory.md`,
  `docs/spec_driven_development.md`, `docs/agentic_review_workflow.md` — keep.

---

## Follow-Ups

### HIGH

- **FU-007 — DDP eps is identical across ranks (confirmed).** `train_generator`
  is seeded with `settings.data_seed` and global RNG with `global_seed`, with no
  rank offset, so both ranks draw the same reparameterization noise each step.
  Files: `src/eqvae/training/selected_runtime_runner.py:815-818`, `:2452-2455`,
  `_train_eps :4018-4047`. Fix: offset the eps generator per rank
  (`data_seed + rank`, or fold rank into a per-step eps seed), keep it seeded;
  coordinate with FU-012 resume; add a gate-health check asserting per-rank
  `eps_abs_mean` differ.
- **FU-008 — best_model.pt selected on rank-0-local shard and wrong view
  (confirmed).** `_validation_best_l1` does `min()` over per-view rows (so it
  tracks the easier `clean` view) and runs before the cross-rank gather.
  Files: `selected_runtime_runner.py:2356-2382`, `_validation_best_l1:2700-2702`,
  `_validation_view_row:2600-2684`. Fix: aggregate per-view L1 across ranks
  (sample-weighted all-reduce; shards uneven with `drop_last=False`), then select
  explicitly on the `deterministic_denoising` view. Mirror in the final/summary path.
- **FU-009 — CURRENT.md self-contradicts (full run launched vs blocked).**
  `CURRENT.md:77-130` says v1 pushed/RUNNING; `:1177-1219` Next-Steps and
  `:1223-1236` Blockers still say selection/debug-gate pending. Fix: rewrite
  Next-Steps to the one live item (approval-gated `status-selected-runtime-full`
  → download → `--verify-full-output`); replace the stale blockers. See D-03/D-04.
- **FU-010 — Two contradictory "selected runtime" values in one doc.**
  `docs/kaggle_cli_workflow.md:222-226` (v3, 14.04 samples/s, 59h) vs `:3-5`/`:564`
  (v5, 27.38 samples/s, ~30h). Fix: delete the v3 block (D-17); confirm no config
  references `dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`.
- **FU-011 — CURRENT.md is ~3076 lines, ~95% narration.** Durable contract is
  buried under `:5-1148` (Active Workstream prose) and `:1247-3053` (verification
  log). Fix: see D-01/D-02; target low hundreds of lines.

### MED

- **FU-002 — KL vs recon balance unverified at runtime.** Mean KL over 16×32×32
  with β-target 1; no magnitude guard. Files: `src/eqvae/losses/vae.py`
  `kl_divergence_loss`; `tests/test_vae_loss.py:26-75`. Fix: (a) add a test
  asserting kl/recon fall in a sane ratio band at β=1; (b) enforce both columns in
  `metrics/train_steps.csv` via `test_selected_runtime_full_run.py`. Watch on the
  live run for posterior collapse / KL domination.
- **FU-003 — β warmup is a raw step-fraction, not pinned to epoch 1.** Files:
  `selected_runtime_runner.py:2456-2461` (`beta_warmup_fraction`), spec0009 config.
  Fix: derive `warmup_steps = optimizer_updates_per_epoch`, or validate
  `beta_warmup_fraction * max_train_steps == 12500` for full runs and record the
  resolved warmup-step count; add a config test.
- **FU-012 — Resume restores rank-0 RNG to all ranks.** Only rank 0 checkpoints;
  all ranks load its RNG. Self-consistent only because of FU-007; fixing FU-007
  without this re-breaks per-rank eps after any resume. Files:
  `src/eqvae/checkpointing.py:251,304-329`; runner `_cuda_rng_checkpoint_state
  :2763-2776`, `_save_checkpoint :870,2306`. Fix: persist per-rank generator/CUDA
  RNG keyed by rank, or re-derive each rank's seed from `(data_seed, rank,
  start_step)`; add a resume test asserting post-resume eps differ across ranks.
- **FU-014 — GOAL.md duplicates/out-stales the spec index.** Per-spec status +
  v5/v6 narration in `GOAL.md:107-152` duplicates and disagrees with
  `docs/specs/README.md:28-34`. Fix: strip status tails, keep durable requirement
  clauses, point to specs/README. See D-20.
- **FU-015 — behavior_inventory_kaggle.md holds live spec-0001 design + stale
  blockers.** `docs/behavior_inventory_kaggle.md:541-649`. Fix: move durable design
  into spec 0001; delete dated slice-status/satisfied-blocker lists. See D-18.
- **FU-016 — Confirm baseline stain corruptor fixed the historical HED orientation
  + uses per-sample/rank/step RNG.** Audit warning at
  `docs/behavior_inventory_kaggle.md:249-255`; cross-check `src/eqvae/corruption/`.
  Fix: verify the HED/OD matrix convention and seeding; mark the audit resolved or
  escalate.
- **FU-017 — No test calls the real validation forward loop.** Only the schedule
  predicate is tested. Files: `_run_scheduled_validation:2565`,
  `_validation_view_row:2600`; `tests/test_selected_runtime_full_run.py:87-91`.
  Fix: synthetic test asserting two view rows, clean view consumes no corruption
  RNG, denoising reproducible, best-selection picks denoising (after FU-008).
- **FU-018 — No telemetry/test for decoder head saturation (out-of-[0,1]).**
  Files: `losses/vae.py` (SSIM clamp), model output head, `_metric_row`/
  `_gate_health_rows :3066`. Fix: add decoder-output RMS + clamp-fraction to
  gate-health and a test asserting it is recorded. (Telemetry side of FU-004.)

### LOW

- **FU-004 — SSIM term clamps out-of-range output invisibly.** `losses/vae.py`
  SSIM term wraps `normalized_to_image_domain`; no final tanh. Fix: keep no-tanh;
  rely on FU-018 telemetry unless the run shows persistent saturation.
- **FU-019 — Train CSV loss/grad metrics are per-rank-local, never reduced.**
  `selected_runtime_runner.py:2331-2340`, grad_norm pre-allreduce. Fix: optionally
  emit a global-mean train loss; at minimum document that CSV scalars are per-rank.
- **FU-020 — AMP-skip / successful_count tracked per-rank with no cross-rank
  assert.** `:2477-2521`, counter `:2302-2303`. Fix: all-reduce/assert agreement
  each boundary to fail fast on DDP desync.
- **FU-021 — Two divergent train-step impls (debug `step.py` vs full
  `_run_train_step`).** Can drift. Fix: share logic or add a parity test; document
  `step.py`/`progress.py` as debug-only.
- **FU-022 — Validation beta uses `target_train_steps`, training uses
  `max_train_steps`.** Differs only under `--dry-run`. Files: `:2456-2461` vs
  `:2641-2646`. Fix: use the same denominator or assert `max==target` outside dry-run.
- **FU-023 — equivariant_vae_transition_plan.md (1005 lines) billed as Active
  Source Of Truth but frames FSQ run as current.** Referenced in GOAL/README/
  AGENTS/CLAUDE. Fix: demote to background, or refresh Status. See D-21.
- **FU-024 — Required paper qualitative figures don't exist yet.** `main.tex:209-214`;
  `figures/` has only placeholders. Fix: track as paper TODOs; generate from
  full-run + evaluator outputs before submission.
- **FU-025 — Paper metrics table is placeholder (keep, fill).** `main.tex:192-207`,
  `tables/` only `.gitkeep`. Fix: keep structure; populate from
  `runs/kaggle/selected_runtime_full_v1` + classifier evidence.
- **FU-027 — Confirm doc-only configs still intended.** `non_eq_vae_baseline.json`,
  `ubc_ocean_masked_holdout_test.json` referenced only in docs. Fix: don't drop;
  note in spec 0001 whether `non_eq_vae_baseline.json` is superseded by the
  selected-runtime-full config.

### PROCESS

- **FU-005 — Spec/doc STATUS blocks must stay short (root cause of the staleness).**
  Authority: `docs/spec_driven_development.md:96-97`, `docs/agentic_review_workflow.md:88-89`.
  Fix: apply the Drop Plan below; add a bullet to `spec_driven_development.md`
  requiring STATUS blocks to hold latest state only.
- **FU-028 — Spec 0008 header says implemented; body narrates v2-v5 as "not a
  passing remote proof".** `docs/specs/0008-...md:3` vs `:242-338`. Fix: see D-09.
- **FU-029 — Specs 0001/0003 + README re-narrate the whole finished benchmark
  saga.** See D-05/D-06/D-08/D-10. Largest single staleness offender.
- **FU-030 — Spec 0009 is the LIVE frontier — do NOT compress to "implemented"
  during cleanup.** `docs/specs/0009-...md:3-9,290-318`. Only normalize formatting;
  update status after the approved status/output verification runs.
- **FU-031 — Benchmark code+tests are FINISHED but load-bearing.** Re-exported by
  `src/eqvae/benchmarking/__init__.py`, imported by tests/CLIs, 7 live
  `kaggle_kernel.sh` subcommands. Fix: retire only as ONE coordinated change
  (kernel + subcommands + CLI + `__init__` + `agent_preflight.sh` + README +
  tests together) recorded in a `docs/decisions/` note. Do NOT git-rm piecemeal.
- **FU-032 — kaggle_cli_workflow.md bakes volatile per-run state + frames retired
  phases as next-steps.** `:13-16,84-86,144-151,438-466,770-788`. Fix: move live
  per-run status to CURRENT.md; reframe retired phases as "retained for reproduction".
- **FU-033 — Apply the same history-drop to the whole landing sequence.**
  `docs/specs/README.md:44-124`, `docs/kaggle_cli_workflow.md` — keep consistent
  with a trimmed CURRENT.md/GOAL.md.
- **FU-034 — Decision 0008 carries finished run outcomes.**
  `docs/decisions/0008-...md:16-20,60-67`. Fix: keep the decision, drop the
  v4-shortlist outcome + trim profile lineage. See D-25.
- **FU-035 — Add a preflight guard against re-committing runs/ artifacts.** After
  D-30, add a check in `scripts/agent_preflight.sh` failing if `git ls-files runs/`
  is non-empty.
- **FU-036 — Paper placeholder-removal pass due before 2026-07-09.** `main.tex`
  scaffold text; large vendor PDFs in `template/IEEEtran/`. Fix: schedule the pass;
  consider trimming the vendor tree to the `.cls`/`.bst` used.
- **FU-037 — Paper Status/Limitations contradicts ground truth.** `main.tex:228-242`
  describes abandoned pretest-v8 state. Fix: rewrite for v5-locked + launched run.
  See D-26.

### ENV

- **FU-006 — /tmp tmpfs `usrquota` + 7.2G cap fills with pytest temp → EDQUOT.**
  Mount still has `usrquota` (clearing /tmp didn't change it), so it recurs.
  Symptom: mass "Disk quota exceeded" test failures + Bash wedging. Fix: when
  tight, `rm -rf /tmp/pytest-of-maximus /tmp/eqvae-python-quality
  /var/tmp/eqvae-python-quality`; route test/run scratch off /tmp. NOT a `runs/`
  problem. Bash tool default timeout is 120s; the gate needs ~5min.
- **FU-038 — overleaf_sync_workflow.md hard-codes project id + absolute paths.**
  `:19-21,66-74`. Single-author repo, leave as-is; parameterize if ever cloned.

---

## Make-It-Current Drop Plan

Delete/compress historical narration so the repo reflects current state. Confidence
HIGH unless noted. Execute, then delete this section's done items.

### CURRENT.md (~3076 lines)
- **D-01 [HIGH]** Drop "Latest Verification" stack `:1247-3053` (~1807 lines of dated
  CI logs). Keep one line: "latest gate: 242 passed, 0 type errors".
- **D-02 [HIGH]** Drop dated handoff backlog in "Active Workstream" from
  "Historical provenance follows." `:131-1148`. Keep the short current-state
  paragraph + live full-run pointer.
- **D-03 [HIGH]** Replace "Next Concrete Steps" `:1177-1219` with the one live item
  (approval-gated status → download → verify).
- **D-04 [HIGH]** Replace stale "Current Blockers" `:1223-1236` with the real one
  (v1 pushed, not yet status-checked/verified). Keep sealed-test-shard + src/nn bullets.

### docs/specs/0001 (~3993 lines)
- **D-05 [HIGH]** Drop status+saga preamble `:3-431` → <10-line Status. Durable
  contract starts at `## Purpose :432`. Keep readiness token literals.
- **D-06 [HIGH]** Drop per-version run outcomes in Training/Config Contract `:~1896-1979`;
  keep eligibility/threshold rules.
- **D-07 [MED]** Drop dated "as of" notes in Verification Commands `:~3319,3352,3373,3598`.

### docs/specs/0003
- **D-08 [HIGH]** Drop v1-v8 remote chronology (Status `:3-86`; body `:~213-440`);
  keep durable guard/metadata rules + the locked-runtime sentence.

### docs/specs/0008
- **D-09 [HIGH]** Drop "Latest Remote Attempt" v2-v5 log `:242-338` → one paragraph
  (v5 passed, durable fixes). Resolves FU-028.

### docs/specs/README.md
- **D-10 [HIGH]** Compress bloated cells `:28,30,33` to 1-2 sentences; drop trailing
  narrative `:44-127` EXCEPT keep guard-phrase list `:38-42`. Don't ADD
  `locked / implementation-ready` or the `kaggle_smoke_ready` backtick phrase to the
  0001 cell (fail-closed guards at kaggle_kernel.sh:521/538). Keep `.md` filename literals.

### docs/kaggle_cli_workflow.md (~951 lines)
- **D-11 [HIGH]** Drop duplicated runtime-selection v1-v6 saga `:196-242` AND `:526-575`
  → one sentence + the dual-T4 selection-gate contract.
- **D-12 [HIGH]** Drop real_data_runtime_pretest v2-v8 saga `:468-524`; keep the
  pretest contract (slug, 300000/30000, no selected_runtime.json write).
- **D-13 [HIGH]** Drop top status banner `:3-30` (per-run narration; lives in CURRENT.md).
- **D-14 [HIGH]** Drop finished-phase post-mortems in "Current State" `:52-129`; keep
  per-kernel contract statements.
- **D-15 [HIGH]** Drop per-version timing logs `:666-768`; keep the cadence rules `:634-664`.
- **D-16 [HIGH/MED]** Compress dated auth-edge narration `:318-388` → durable OAuth
  mechanism + "check UI for quota". (Quota snapshot line MED.)
- **D-17 [MED]** Delete the superseded v3 selected-runtime block `:222-226` (the FU-010
  hazard); compress v8 provenance `:153-195,247-282`.

### docs/behavior_inventory_kaggle.md (~648 lines)
- **D-18 [MED]** Move/drop "Spec 0001 Reopened Decisions" `:541-649` (wrong home; FU-015).
- **D-19 [MED]** Drop time-stamped operational asides + 403 note `:109-114`; keep tables.

### GOAL.md
- **D-20 [MED]** Strip status narration from "Do Not Lose These Requirements" `:107-152`;
  keep durable clauses, point to specs/README.
- **D-21 [LOW]** Demote/refresh transition-plan billing in "Active Source Of Truth" `:24-26`.

### docs/specs/0009 (LIVE — formatting only)
- **D-22 [HIGH]** Drop dated local-verification log `:243-267`; keep command list `:229-241`.
- **D-23 [HIGH]** Drop adversarial-review blow-by-blow `:269-288`.
- **D-24 [MED]** Drop dated approval/run-status annotations `:290-307,309-318`; keep the
  command block + the "awaiting approved status/output" blocker.

### docs/decisions/0008
- **D-25 [MED]** Drop v4-shortlist amendment `:60-67` + trim profile lineage `:16-20`.

### paper/sipaim2026
- **D-26 [HIGH]** Rewrite stale Status/Limitations `main.tex:228-242` (FU-037).
- **D-27 [HIGH]** Replace Spanish/cat thesis figure `transformaciones_gato_wsi.png`
  (`main.tex:113`).
- **D-28 [MED]** Redraw/replace `semi_supervised_model.png` (`main.tex:79`).
- **D-29 [MED]** gitignore latexmk byproducts (`main.bbl/.fdb_latexmk/.fls` not ignored).

### Tracked artifacts / orphans
- **D-30 [HIGH]** `git rm --cached` the 4 tracked files under
  `runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark/` (32K; only tracked files
  under ignored `runs/`). Pair with FU-035.
- **D-31 [HIGH]** Delete root `Screenshot from 2026-01-30 14-43-57.png` (123 KB,
  zero references repo-wide).
