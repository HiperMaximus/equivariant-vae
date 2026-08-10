# Spec 0011: Kaggle Training-Configuration Search

Status: v5 lean contract / runtime selected / baseline session 1 running

Last updated: 2026-08-10

## Decision

Spec 0011 exists to choose the training configuration used by the two paper models on
Kaggle dual T4. It is a one-off benchmark, not a reusable optimizer, controller
architecture, or audit-grade experiment platform.

The previous v4 coverage/certificate/controller contract is superseded. Do not rebuild
its exhaustive installed-source inventory, independent duplicate verifier, generalized
transformation DAG, capsule-equivalence protocol, or mutation-resistant evidence system.
Those mechanisms do not materially improve the configuration chosen for these two runs.

Neutral, redundant, or inert options may remain enabled. There is no objective to find
the smallest recipe or explain which individual toggle caused a speedup.

## Goal

For correct dual-T4 `drop_last=True` training, choose the measured recipe and integer
global batch that minimize

```text
floor(real_train_patch_count / global_batch)
* synchronized_mean_steady_state_step_wall_time
```

The output is one concrete configuration for the non-equivariant VAE. Repeat the compact
search or confirm compatibility for the continuous-`SO(2)` architecture before its run.

Do not select by step latency alone, throughput alone, largest feasible batch, compile
time, fewest toggles, or an assumed monotone memory ceiling.

## Fixed experiment boundary

- Kaggle script kernel, two visible T4 GPUs, one process per GPU, NCCL DDP.
- Latest PyPI Torch available when the campaign starts, installed before project import.
- The actual repository model, optimizer, AMP path, corruption, clipping, zero-grad,
  forward/backward/update body, and `drop_last=True` semantics used by training.
- Generated device-resident inputs are acceptable for initial step-body screening.
  Finalists must be checked with the real loader/training path before promotion.
- Compile/startup time is excluded. Measurements begin only after compile and allocator
  behavior settle.
- Global batch equals the sum of the two equal per-rank batches.

## Candidate recipes

Use a concise, reviewed list assembled from:

1. repository fast paths and the successful FSQ training reference;
2. public options exposed by the installed Torch APIs used by this training path;
3. a small number of internal/experimental options with a concrete mechanism relevant
   to dual T4 DDP, Inductor, memory layout, optimizer, AMP, or communication overlap;
4. the immutable 309-row v2 results as ordering, failure, VRAM, timing, and batch priors.

The list must include the eager correctness control and plausible complete bundles for:

- `torch.compile` mode and DDP optimization/compiled-autograd combinations;
- contiguous versus channels-last where supported;
- DDP static graph, gradient bucket views, bucket size, and useful communication hooks;
- fused optimizer, foreach clipping/zero-grad behavior, AMP/autocast cache;
- cuDNN benchmark, TF32, and matmul precision;
- repository fast paths that execute in the measured training body.

Do not mechanically enumerate every internal ConfigModule value, environment variable,
or require formal all-pairs coverage. Use current official documentation or a tiny Kaggle
probe only when a concrete API/runtime uncertainty blocks the next measurement.

Recipes are complete bundles. Prefer broad compatible bundles and a few important direct
interaction variants over singleton toggle experiments. Do not remove a working option
merely because it may be inert. Subtraction is warranted only to fix correctness,
compilation, memory, or a material performance regression.

## Recipe and activation rules

Each recipe has a stable ID derived from all requested settings. The worker records the
requested settings and the relevant effective readbacks available from public/runtime
state. A recipe is invalid when:

- the process crashes or compilation cannot settle after one retry;
- the requested setting is rejected or a required fast path demonstrably did not run;
- either rank is missing, on the wrong device, or reports non-finite loss/gradients;
- parameters do not update, ranks disagree on synchronized parameters, or required
  compile activity is absent;
- the run violates the fixed model/training contract.

An unobservable neutral toggle does not invalidate a recipe when its surrounding recipe
is correct and the toggle is not required for the claimed fast path.

## Joint recipe and batch search

Search `(complete_recipe, per_rank_batch)` coordinates directly with a short interactive
Kaggle sequence:

- Probe the current recipes at a high batch (initially 100).
- Use a small binary search to bracket each recipe's practical capacity.
- Assuming one broad objective well, use ternary/golden-section refinement between a low
  batch and the feasible upper end.
- Check the winning integer batch and its immediate neighbors once.
- Structured OOM invalidates only that coordinate; observed irregularity may justify one
  extra spot check, not a full matrix.
- Old v2 observations guide the bracket and recipe order but do not prove the upgraded
  runtime.

For each successful rank, retain synchronized block wall time after settle. Coordinate
time is the slower-rank mean. Compute the objective using the real train-patch count and
integer global batch exactly as shown above.

## Measurement and confirmation

Each selectable coordinate runs in a fresh worker process. Diagnostic reuse inside a
worker is allowed only for scouts that cannot become selectable results.

Screening requirements:

- two successful settle/update cycles before timing;
- at least 20 timed optimizer updates unless a focused pilot establishes that a longer
  block is required for stable ranking;
- synchronized rank timing with host synchronizations outside the timed train body;
- peak allocated/reserved VRAM and compile counters where applicable;
- atomic append/checkpoint after every attempted coordinate.

Repeat the apparent winner two or three times. If the difference from the runner-up is
smaller than ordinary run-to-run noise, keep the lower-VRAM/stabler recipe. Do not spend
hours resolving a tie that cannot materially change total training time.

## Minimal persistence

During the campaign, each direct run wrote a plain CSV plus summary JSON. Downloaded raw
results remain ignored under `runs/kaggle/`; the compact durable outcome is
`configs/spec0001/non_eq_vae_runtime_winner.json`. The one-use probe kernel and wrapper
were deleted after selection. Do not rebuild them unless the runtime changes and a new
measurement question actually requires it.

## Output

The completed campaign retains the selected concrete recipe/batch, timing confirmations,
VRAM, and raw-evidence locations in the winner JSON. Translation into the runner plan is a
separate step because the historical v5 selection artifact must not be relabeled.

The translated plan lives at
`configs/spec0001/non_eq_vae_selected_runtime.json`. It is a compact consumer config,
not a regenerated benchmark artifact: it hash-links the immutable winner JSON, carries
only settings the existing runner applies, derives `6000` updates/epoch from global batch
50, and retains explicit launch blockers until the real LR/loader/overfit gate passes.
The historical `runs/kaggle/runtime_selection_v5` tree remains byte-unchanged.

## Real-data learning gates

Runtime promotion requires two distinct bounded checks on the actual UBC train surface:

1. **LR range.** One dual-T4 run uses the selected batch-25 compiled recipe and increases
   the base AdamW learning rate exponentially over 192 successful updates from `2e-5` to
   `3e-3`. It writes per-step LR/loss rows and a compact summary containing instability,
   the minimum smoothed loss, the steepest descending log-LR segment, and a conservative
   recommended LR. Non-finite loss/gradients, an AMP skip, missing rank, or unapplied
   runtime fails the result. This is a bounded decision aid, not a reusable tuner.
2. **Tiny overfit.** The existing canonical fixed-32 real-train selector is repeated only
   at the sampler to full batch-25 microbatches (50 global samples/epoch). Over 128
   successful updates, both smoothed L1 and smoothed reconstruction loss must improve by
   at least 5%, with two ranks, finite metrics, zero AMP skips, passing gate health, and
   the translated runtime proved applied. This answers whether the network/optimizer can
   learn before the full run; it is separate from LR selection and not convergence
   evidence.

The passing LR range selected effective `7.216878e-4` for the tiny proof. Full training
uses a 600-update linear warmup to effective `1e-3`, then cosine decay without restarts to
`1e-5` at update 60000. The shared model-base LR remains unchanged. Tiny overfit uses a
10-update linear warmup to `7.216878e-4`, then holds it constant through update 128.

## Acceptance criteria

1. The exact floored epoch objective, not Bmax or step latency alone, selected the winner.
2. The Kaggle measurements prove two T4 ranks, correct device assignment, the update body,
   finite synchronized updates, effective core settings, settle behavior, and slower-rank
   timing.
3. Neighbor batches and structured OOM coordinates inform selection without assuming a
   monotone capacity boundary.
4. Two fresh measurements support the selected coordinate.
5. The 192-update real-data LR range passes and records the frozen LR decision.
6. The real-loader 8-update resume/debug check and 128-update fixed-32 overfit check pass;
   the overfit check requires at least 5% smoothed L1 and reconstruction improvement.
7. Focused tests, Ruff, BasedPyright, `./scripts/python_quality.sh`, `git diff --check`,
   repo preflight, workspace preflight, and one clean-context adversarial review pass.
8. No Kaggle push/run occurs without explicit remote-write authorization and
   `KAGGLE_PUSH_CONFIRMED=1`.
9. The matched baseline and continuous-`SO(2)` full runs use beta target `0.01`, selected
   by the paired fixed-32 v10 probe, with the existing one-epoch full-run beta ramp.
10. Full-run publication pins the measured Torch/CUDA stack and beta target, verifies
    the entire generated wrapper against its source template, and publishes each
    checkpoint atomically before hashing it.

## Implementation state

Non-equivariant selection completed on 2026-08-09: Python-reducer compiled whole-step
FP16 channels-last at per-rank batch 25
(global 50). Two fresh measurements project 4499 and 4672 seconds per 300,000-patch
epoch; batches 18, 35, and 56 were slower. The exact recipe remains in
`configs/spec0001/non_eq_vae_runtime_winner.json`; probe artifacts are under
`runs/kaggle/runtime_recipe_probe_v9` and `runtime_recipe_probe_v14`.

The synthetic/device-resident timing campaign is complete; its one-use code was removed
and no more batch probes should run.
Kaggle LR-range v1 and debug/tiny v6 passed on 2026-08-09. The range completed 192
two-rank updates with zero skips/non-finites. The resume probe continued update 4 through
8. Fixed-32 overfit completed 128 two-rank updates; smoothed L1 improved 28.1% and
reconstruction loss 22.5%, both above the 5% gate. The network can learn under the
selected runtime. A stricter one-off clean probe (`beta=0`, deterministic latent,
zero corruption) then completed 512 updates in Kaggle v8: smoothed L1 improved 61.5%
(`0.2416 -> 0.0930`) and was still descending in the final block, with zero skips or
non-finites. The 1024-update clean Kaggle v9 probe improved smoothed L1 68.9%
(`0.2534 -> 0.0788`) and reconstruction loss 61.5% (`0.3194 -> 0.1230`) with zero
skips/non-finites. All 64-step bins descended, but deterministic clean fixed-training
metrics missed the deliberately strict L1-below-0.05/80% target. The saved image artifact
is from held-out validation, not the fixed train set, so it is not memorization evidence.
The one-off clean kernel branch was removed. Kaggle v10 completed independent slow-ramp
beta `0.01` and `0.1` probes with the same seed. Post-training fixed-32 clean
`model.eval()`, `z=mu`, beta-zero evaluation gave reconstruction/L1/SSIM/unweighted-KL
`0.11872/0.07748/0.58764/0.47441` at beta `0.01` and
`0.14724/0.09708/0.49837/0.10411` at beta `0.1`, with zero skips/non-finites. Beta `0.1`
therefore compresses more but materially harms retained image information. The user
locked beta `0.01` for the matched baseline/continuous-`SO(2)` comparison; do not run an
intermediate beta probe. The full config now overrides the old model-base beta-`1.0`
default with target `0.01` while retaining the one-epoch beta ramp. The completed
paired-probe wiring is removed and the dedicated full kernel passes local preflight. The
lean session
contract is: every worker targets update 60000 and runs until it finishes or Kaggle
closes it; every 3000 updates atomically flush the evaluation/artifact boundary and
checkpoint; download each session separately; give the next session only the latest
fully completed checkpoint. The checkpoint recorded and hashed in
`benchmark/checkpoint_resume_proof.json` is the session commit point. For final
concatenation, retain only CSV rows and fixed-25 boundary artifacts at or below its
`latest_checkpoint_step`; a hard close can leave preflushed rows above that step, and
those rows are uncommitted. Full resume rejects checkpoints outside the 3000-step
boundary schedule. Resume must
skip old data by sampler indices rather than rereading old patch payloads, and must
re-separate stochastic streams per DDP rank after loading the rank-0 checkpoint. Do not
add an artificial session cap, remote artifact-tree transport, a generalized session
manager/merge service, or automated cleanup. Run focused/broad gates and a clean launch
review, then request separate approval for the guarded session-1 baseline push. Before
session 2, attach only the concrete latest checkpoint downloaded from session 1; the
current metadata/push guard still permits only the UBC dataset and therefore cannot yet
expose that checkpoint to a fresh worker.
Session-1 local review additionally pins the beta target and measured Torch
`2.13.0+cu130` / CUDA `13.0` stack at launch, verifies the complete ignored wrapper
against its tracked template, and publishes checkpoints through a temporary sibling and
atomic rename before recording their hashes. The saved state covers the model,
optimizer, AMP scaler, CPU/CUDA/NumPy and explicit generator RNG, absolute successful
update, DDP sampler progress, and best metric. LR and beta schedules derive from the
absolute successful-update count. This is the smallest experiment-specific continuation
contract; no generalized checkpoint/session framework is planned.
Baseline session 1 launched with explicit approval on 2026-08-10 as Kaggle kernel
`maximusshtefan/eqvae-selected-runtime-full`, version 2, from GitHub source commit
`81b5017`. The guarded API check and push passed; the single handoff status read was
`KernelWorkerStatus.RUNNING`. Do not continuously poll or rerun it without new direction.
Downstream probes remain the final compression-utility criterion. Repeat only the lean
architecture-specific tuning required for continuous `SO(2)` later; do not reopen the
shared beta choice by default.
