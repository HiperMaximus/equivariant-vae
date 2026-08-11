# Current Repository Status

Last updated: 2026-08-11

## Fresh-session start here

Read `AGENTS.md`, `GOAL.md`, this file, `docs/specs/README.md`, and active Spec
0011 completely. Baseline full-run session 1 is Kaggle kernel version 2 from source
commit `81b5017`; it ended `KernelWorkerStatus.ERROR` after completing the 15000-update
boundary. Its output and checkpoint are verified locally; the FSQ-aligned AMP runtime
and full-output verification corrections are implemented and verified. Session 2 was
Kaggle kernel version 3 from clean source commit `65112aa`; Kaggle closed it with
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` after the complete update-45000/epoch-7.5
boundary. Its output is downloaded and the checkpoint/proof prefix is verified locally.
Final session 3 is Kaggle kernel version 4 from clean source commit `462c9e1`; it reached
`KernelWorkerStatus.COMPLETE` at update 60000/epoch 10.0 from the exact update-45000
checkpoint. Its remote output is not yet downloaded or locally verified.
Preserve any later unrelated or ambiguous work: do not reset, checkout, blanket-restore,
or recreate the tree. Inspect every diff before surgical removal.

Non-equivariant runtime selection is complete. Do not execute old-v2 `p00310` or the
failed-v3 Bmax/main-effects controller. Session-1 source commit `81b5017` and the first
AMP runtime correction are pushed to GitHub; the current HEAD includes the full-output
policy correction.

Baseline full-run session 1 ran on Kaggle as
`maximusshtefan/eqvae-selected-runtime-full`, kernel version 2. The user explicitly
approved its push on 2026-08-10; the guarded API check and push passed. The terminal
status is `KernelWorkerStatus.ERROR`. Logs show successful boundary evaluation and
checkpoint completion at updates 3000, 6000, 9000, 12000, and 15000, followed by an
intentional failure on both ranks when the AMP guard detected an overflow in the
deferred-metrics window ending at update 18000. Output is downloaded under ignored
`runs/kaggle/selected_runtime_full_v2`. Its resume proof names update 15000 and SHA-256
`8f1b2af601354642036d4d71dca8865ea9c7896a71da4ed69f3871559c448f4f`; the local
checkpoint hash matches exactly. Its committed CSV prefix and fixed-25 boundaries also
end at 15000. The fixed-25 output includes originals, reconstruction progress, rotated
inputs and latents at 90/180/270 degrees, error maps, deterministic posterior-`mu`
arrays, first-three-channel views, and PCA views.

The failure was stricter than FSQ, not checkpoint corruption or missing norm clipping.
Both real runner paths passed `observe_skip=False`, so GradScaler skipped/backed off but
the hot loop could not gate successful-step progress immediately; a deferred scale-drop
assertion then killed the run. The update-15000 checkpoint shows scale `1048576`, growth
factor `2`, backoff factor `0.5`, and growth interval `2000`, consistent with normal
dynamic-scale growth/backoff. Gradient clipping was already global norm `1.0` with
foreach. The downloaded rows show one actual two-rank overflow at event label 14007; the
old path let its data/schedule label advance even though GradScaler discarded that
physical optimizer update. The update-15000 checkpoint is finite and loadable but
therefore represents 14999 physical optimizer updates; the user explicitly accepts losing
one or a few updates. The fix sets `observe_skip=True` on eager and compiled paths,
removes the deferred fatal assertion, consumes the next batch after a skip, and does not
advance LR, beta, validation, checkpoint, or successful-update counters. Full-run
summaries and the downloaded-output verifier allow non-finite telemetry only on skipped
rows and still require every successful row to be finite with exact step/rank coverage.
Runtime selection, LR-range, debug, and tiny-overfit gates remain strict zero-skip.
The user locked beta `0.01` on 2026-08-09; do not run an intermediate beta probe.

## Current objective

Spec 0011 is now a lean, two-architecture Kaggle tuning campaign. For correct dual-T4
`drop_last=True` training, minimize

```text
floor(real_train_patch_count / global_batch)
* synchronized_mean_steady_state_step_wall_time
```

Recipe and integer batch are selected jointly. Largest feasible batch, step latency,
throughput alone, compile time, and the fewest enabled options are not objectives. A fast
correct recipe may retain neutral, redundant, or inert toggles.

## Direction correction

The user rejected the uncommitted Spec 0011 v4 implementation because it had become an
audit-grade general platform for a configuration search that will be used for only the
non-equivariant and continuous-`SO(2)` models. The active spec was replaced by v5 on
2026-08-08.

V5 keeps only what can change or establish the selected training configuration:

- reviewed complete recipe bundles from repository/Torch fast paths;
- joint nonmonotone recipe×batch measurement;
- correct dual-rank update/timing checks;
- coordinate-local OOM and focused failure repair;
- one CSV and summary JSON per direct Kaggle probe;
- fresh finalist confirmation and a real-loader check;
- a concrete selected-runtime config fragment.

It drops exhaustive internal/source inventory, formal all-pairs coverage, independent
duplicate verification, generalized transformation DAGs, capsule/cache equivalence
certificates, and broad audit-mutation machinery. Neutral options do not need ablation.

## Immutable v2 evidence

`docs/data/spec0011_runtime_recipe_v2/` retains 309 immutable rows and its producer. Use
the rows as explicit priors for option ordering, failure modes, VRAM, timing regions, and
batch wells. The incomplete `p00310` is permanently unschedulable. Old rows do not prove
performance under a newly upgraded runtime.

## Rollback boundary

Proven useful and preserved:

- existing training/fastpath and selected-runtime work;
- latest-PyPI Torch upgrade support;
- immutable v2 evidence;
- guarded Kaggle packaging and atomic publication foundations;
- unrelated dirty repository changes.

Removed as v4-only overbuild:

- the activation/controller/identity/inventory/measurement/independent-verifier stack;
- the v4 maximal-cover/statistical policy and audit-focused tests;
- v4 artifact-parent, certificate, and executor-readiness packaging assumptions.

The large failed-v3 controller and the later one-use direct probe kernel are removed. Do
not restore their exact-Bmax, main-effects, beam/frontier, exhaustive-audit, certificate,
or bespoke packaging behavior. The measured winner is retained as one compact config.

## Selected non-equivariant runtime

Use per-rank batch 25 (global 50), conservative FP16 AMP, channels-last, compiled whole
step, Python DDP reducer, compiled autograd, compute/communication reorder, fused AdamW,
TF32, high matmul precision, gradient-as-bucket-view, bucket cap 50 MB, no buffer
broadcast, and foreach clipping. Exact recipe and compact measurement pointers live in
`configs/spec0001/non_eq_vae_runtime_winner.json`.

Fresh batch-25 measurements were 749.8 and 778.7 ms/step, projecting 4499 and 4672 s per
300,000-patch epoch; their mean is about 4585 s (1.27 h). Neighbor batches 18 and 35
projected 4719 and 4954 s; batch 56 projected 5166 s. Baseline AMP projected 10155 s.
VRAM reserved was 6078-6104 MB at the selected coordinate. All selected rows had finite
updates, synchronized ranks, zero AMP skips, zero graph breaks/recompiles after settle,
and zero measured data-wait fraction. Kaggle kernel version 14 is the final confirmation;
do not run more batch probes.

Raw downloaded rows remain ignored under `runs/kaggle/runtime_recipe_probe_v9` and
`runtime_recipe_probe_v14`. The one-use `runtime_recipe_bakeoff` module, Kaggle kernel,
CLI actions, guards, generated launcher, and focused tests were removed. Only the compact
winner JSON and immutable v2 evidence remain.

Exact removed probe surfaces: `src/eqvae/benchmarking/runtime_recipe_bakeoff.py`,
`tests/test_runtime_recipe_bakeoff.py`, `kaggle/kernels/runtime_recipe_bakeoff/`, and all
of their dedicated wiring in `scripts/kaggle_kernel.sh`,
`scripts/build_kaggle_embedded_kernel.py`, `scripts/agent_preflight.sh`, `.gitignore`, and
`tests/test_kaggle_embedded_kernel.py`. The generic latest-Torch policy test remains.

## Multi-session handoff

- Session-2 transport is complete. Private dataset
  `maximusshtefan/eqvae-baseline-session1-step15000` contains only the verified
  `step_015000.pt`; the kernel attaches it beside the UBC patch dataset and rejects a
  missing or hash-mismatched checkpoint before GPU work. Keep this one-off transport;
  do not build a generic layer.
- Use lean checkpoint-only sessions because the projected ~12.7-hour training time
  exceeds Kaggle's 8-hour limit. Every session still targets update 60000 and runs until
  it completes or Kaggle closes it; there is no artificial session cap. Every 3000-update
  boundary flushes metrics, fixed-25 evaluation artifacts, and a resumable checkpoint.
  After cancellation, download the whole session into its own local directory and give
  the next Kaggle worker only the latest fully completed boundary checkpoint. Resume in
  a fresh output directory with index-only loader offsetting and rank-local stochastic
  streams. The checkpoint named and hashed in
  `benchmark/checkpoint_resume_proof.json` is the session commit point. Use
  `latest_checkpoint_step` as the inclusive cutoff; exclude any preflushed CSV rows or
  fixed-25 artifacts above it, and reject a missing/hash-mismatched/non-3000 boundary.
  After update 60000, concatenate the committed CSV prefixes locally by absolute
  optimizer step and choose the global best from downloaded validation results. Keep
  session copies until the merged result is verified; delete redundant copies only
  afterward.
- Session 1 version 2 was pushed from clean commit `81b5017` and ended in error after the
  completed 15000 boundary. Its output is downloaded and verified; do not replace or
  rerun it without new explicit direction.
- Session 2 version 3 was pushed from clean commit `65112aa` at 2026-08-10 11:24 COT and
  ended `CANCEL_ACKNOWLEDGED` after the completed update-45000 boundary. Its output is
  downloaded under ignored `runs/kaggle/selected_runtime_full_v3_session2`. The resume
  proof names `checkpoints/step_045000.pt` with SHA-256
  `e7a0f05e013bff4f7a5bfbfd4442f3c9a6d19cf261c42f54a6d04391be76e88b`; the local
  file hash and loadable schema-v5 metadata match. Fixed-25/validation boundaries are
  complete from 18000 through 45000. The session logged 26 skip rows: 13 synchronized
  two-rank AMP skips, all isolated (maximum consecutive streak one), with zero non-finite
  successful updates. Clean validation L1 improved `0.07733 -> 0.05965` and denoising
  L1 `0.08172 -> 0.06279` from update 3000 through 45000; corresponding SSIM rose
  `0.5690 -> 0.7272` and `0.5647 -> 0.7192`. Visual comparison of the same fixed-25
  originals at updates 3000, 15000, and 45000 shows continued edge/nuclear-contrast
  improvement without collapse: tissue layout, stain, glands, nuclei, stroma, and empty
  regions remain recognizable, with expected VAE smoothing of fine chromatin detail.
  Session-1/session-2 `originals.png` and `originals.pt` hashes match exactly, proving the
  comparison did not replace the fixed examples. The two session output directories are
  separate; do not overwrite either when preparing session 3.
- The current local source adds FSQ-aligned rank-0 resume breadcrumbs for data preparation,
  checkpoint-load start, load duration/restored epoch-step/LR/GradScaler scale, and the
  first successful resumed optimizer update with LR/attempt count. It also logs every
  rare AMP non-finite/overflow skip with its consecutive streak and new scaler scale,
  then logs recovery. NCCL initialization also receives the resolved rank-local CUDA
  device explicitly, removing PyTorch 2.13's ambiguous-barrier-device warning. These
  apply to session 3; version 3 predates them.

The user prefers direct, bounded Kaggle experiments over defensive local machinery and is
comfortable with liberal probe pushes. Still use the repository's `KAGGLE_*_CONFIRMED`
guards; never infer permission for the full training launch from probe permission.

`configs/spec0001/non_eq_vae_selected_runtime.json` is a hash-linked runtime snapshot.
Its `full_training_launch_ready=false` and probe-era blocker strings record creation-time
state; do not casually edit them and invalidate downloaded runtime/debug evidence. Live
readiness and the remaining multi-session blocker are recorded here.

## Verification state

Before the direction correction, the overbuilt implementation passed its focused tests,
the full 768-test Python quality gate, `git diff --check`, and both preflights. Those
results do not verify the replacement. Rerun focused tests and every required gate after
the rollback and lean implementation.

Kaggle LR-range kernel v1 passed: 192/192 two-rank updates from `2e-5` to `3e-3`, zero
AMP skips/non-finites, and smoothed loss `0.645 -> 0.251`. Manual curve inspection chose
effective `7.216878e-4` for tiny overfit and peak `1e-3` for scheduled full training; the
automatic `2e-5` recommendation was rejected as startup-noise-biased.

Kaggle debug/tiny kernel v6 passed. Resume loaded update 4 and continued to update 8.
The fixed-32 check completed exactly 128 updates on both ranks with zero skips/non-finites;
smoothed L1 improved `0.2388 -> 0.1717` (28.1%) and reconstruction loss improved
`0.3048 -> 0.2361` (22.5%). The strict downloaded-output verifier passed.

The stricter clean-memorization probe removes corruption, latent sampling, and KL
(`beta=0`, deterministic `z=mu`) while retaining the selected runtime/LR. Kaggle v7
stopped before training because the zero corruption probability did not match the named
locked profile; the direct fix added the explicit `no_corruption_probe` profile. V8 then
completed 512 two-rank updates with zero skips/non-finites. Smoothed L1 improved
`0.2416 -> 0.0930` (61.5%) and reconstruction loss `0.3076 -> 0.1451` (52.8%); the final
64-step block still descended, so the network clearly learns but has not met the stricter
memorization target of 80% L1 reduction or L1 below 0.05. Evidence is under
`runs/kaggle/selected_runtime_clean_memorization_v8`. Its legacy tiny-summary status is
expectedly `fail` only because that old gate requires exactly 128 updates; the v8 kernel
itself completed and the raw 512-step evidence is valid.

The same clean probe ran for 1024 updates as Kaggle kernel v9. It completed 1024 two-rank
updates with zero skips/non-finites; smoothed L1 improved `0.2534 -> 0.0788` (68.9%) and
reconstruction loss `0.3194 -> 0.1230` (61.5%). Every 64-step bin improved and the last
step reached L1 `0.0780`, but the run did not meet the deliberately strict 80% reduction
or L1-below-0.05 target. These are deterministic clean fixed-training-batch measurements.
The saved image artifact is instead a clean held-out validation sample and must not be
used to diagnose fixed-set memorization. Evidence is under
`runs/kaggle/selected_runtime_clean_memorization_v9`.
The one-off kernel branch was removed after the run; the compact probe config/evidence
contract remains. Kaggle v10 then completed the paired regularized fixed-32 probes with
the same seed, 512-step beta ramp, and 1024 updates. In the post-training clean
`model.eval()`, `z=mu`, beta-zero evaluation, final beta `0.01` produced reconstruction
loss `0.11872`, L1 `0.07748`, SSIM `0.58764`, and unweighted KL `0.47441`; final beta
`0.1` produced `0.14724`, `0.09708`, `0.49837`, and `0.10411`, respectively. Both runs
completed with zero skips/non-finites. Beta `0.1` compresses more but materially degrades
the retained image information, so it is not the default candidate. Evidence is under
`runs/kaggle/selected_runtime_beta_probe_v10`. The original beta-1 run drove KL
effectively to zero.

Latest local verification after the resume/AMP observability addition: the focused
full-run suite passes 216 tests, the dedicated full-kernel preflight passes 217 tests, and
`./scripts/python_quality.sh` passes formatting, Ruff, 687 tests with 1 skip, and
BasedPyright with 0 errors. The repo/workspace preflights and `git diff --check` pass.
Earlier clean-context audits found no launch blocker in
the checkpoint/session/runtime slice. They
confirmed atomic checkpoint publication, the hashed 3000-step checkpoint as the session
commit point, index-only loader continuation, rank/segment-separated stochastic streams,
fixed-25 completion before checkpoint commitment, exact generated-wrapper verification,
beta `0.01`, and the measured Torch `2.13.0+cu130` / CUDA `13.0` stack. The checkpoint
state was also cross-checked against `kaggle/fsq_train_reference.py`: model, optimizer,
scaler, RNG, progress, and best metric are covered; LR/beta progress derives from the
absolute successful-update count. Source commit `81b5017` is on GitHub. Kaggle session 1
version 2 ended `ERROR`: logs show completed boundaries through update 15000 and an AMP
overflow guard failure in the window ending at update 18000. Its downloaded proof,
checkpoint hash, metric prefix, and fixed-25 boundaries verify update 15000 as the commit
point. Session 2 then advanced the committed prefix through update 45000; its exact
checkpoint hash, loadable state, finite successful rows, validation curve, fixed-25
boundaries, and partial manifest are verified locally.

## Fresh-agent execution order

1. Keep beta `0.01` locked for the matched baseline/continuous-`SO(2)` comparison; beta
   `0.1` is rejected and no intermediate beta probe is planned. Downstream performance
   will evaluate usefulness, not reopen beta tuning by default.
2. Keep the multi-session implementation deliberately small: let Kaggle close an
   unfinished worker, resume from its latest complete 3000-step checkpoint, and
   concatenate downloaded CSVs locally after completion. Do not build an artificial
   session cap, remote artifact-tree transport, generalized session manager, merge
   service, or cleanup framework.
3. Treat paired-probe removal, index-only resume, per-rank RNG rebasing, atomic
   checkpoint publication, focused tests, full quality, full-kernel preflight,
   repo/workspace preflights, and the post-fix clean-context audits as complete.
4. Treat source commit `81b5017`, GitHub push, guarded API check, Kaggle kernel version 2
   push, and terminal `ERROR` status/log reads as complete. Do not continuously poll.
5. Treat the session-1 output download, update-15000 proof/hash/CSV/fixed-25 verification,
   FSQ comparison, FSQ-aligned AMP runtime/full-output correction, full quality gate, and
   full-kernel preflight as complete. The checkpoint is accepted with one lost physical
   optimizer update under the user's stated tolerance.
6. Treat the exact session-1 checkpoint upload, session-2 path/hash guard, clean commit
   `65112aa`, GitHub push, version-3 launch, terminal cancellation, output download, and
   committed update-45000 verification as complete.
7. Treat the smallest one-off session-3 transport as locally ready at ignored
   `runs/kaggle/session2_resume_dataset`: it contains only dataset metadata and the
   verified update-45000 checkpoint. Final-run metadata, wrapper, and launch guards are
   pinned to private slug `maximusshtefan/eqvae-baseline-session2-step45000`, exact
   Kaggle path, and SHA-256. Do not build a generic session manager.
8. Treat the private checkpoint-dataset upload, source commit `462c9e1` GitHub push,
   guarded final-kernel push, and Kaggle version-4 execution as complete. Remote logs
   show completed boundaries 48000, 51000, 54000, 57000, and 60000, with six isolated
   AMP skips that each recovered on the following attempt. The remote output lists
   `step_060000.pt`, `final.pt`, summaries, manifests, and fixed-25 artifacts.
9. Download version-4 output into a new ignored session-3 directory, never either prior
   session directory; run the strict verifier, validate the final checkpoint/proof and
   concatenated metrics, then inspect update-60000 reconstructions and rotated artifacts.

Baseline full training reached all 60000 successful updates; local final-output
verification remains. The continuous-`SO(2)` repeat remains a later gate.
