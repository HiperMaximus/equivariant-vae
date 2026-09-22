# Current Repository Status

Last updated: 2026-09-22

## Active frontier

[Spec 0053](docs/specs/0053-functional-riemannian-latent-geometry.md) is the
single live contract for comparing the functional latent geometry of the frozen
normal and continuous-`SO(2)` VAEs. Stage A1 is accepted; Stage A2 is
`draft active`. Calibration v1 and v2 completed `unresolved`; Kaggle v3 failed
during input startup and v4 on an unnecessary compile gate. Simplified v5
completed successfully and resolved the search-space question: `d=128` is
conditioned but materially restricts the optimized paths relative to direct
full-latent control. The replacement-account full-latent calibration completed
successfully and fixed the original Stage A2 solver at `K=32`, at most 512 Adam
steps and best-iterate retention. That result remains frozen, but the completed
post-hoc Stage A2b run shows that its paths were solver-limited. The compact
Stage A2 numerical contract constructs `U` deterministically from the first
full-latent side, uses one common FP32-aware SVD convention, and specifies a
one-shot shooting consistency IVP plus fixed time and frozen-path quadrature
refinements. The scientific runner and its explicit thin Kaggle entrypoint are
implemented locally. Stage A2 numerical computation completed in Kaggle v5;
the recovered per-model outputs and locally assembled combined result show that
all 32 sides and 12 bridges attained their recorded best energy at the 512-step
ceiling. Stage A2b continued 12 informative full-latent paths for 1,024 more
steps. Every path again attained its best value at the final step and no LR
plateau reduction fired, so exact continuation from the newly saved optimizer
state is the active numerical next step. Scientific residuals remain continuous
measurements rather than comparison tolerances, acceptance gates, or runtime
errors; only numerical singularity limits which intrinsic quantities are
defined.
The local Kaggle CLI is authenticated as replacement account
`maximusshtefan`. Private dataset
`maximusshtefan/eqvae-frozen-vae-weights-v1/1` contains only the two accepted
state files; a clean redownload verified their immutable SHA-256 values
`30064fa...87c7` and `06802ceb...3c12`. The active kernel metadata and
owner-independent contract use that dataset. No active runner or contract
depends on `maximshtefan`.

An anonymous six-page INCISCOS 2026 manuscript now lives under
`paper/inciscos2026/`. It uses the IEEE A4 conference format and synthesizes
only accepted evidence through Stage A1; the active Stage A2 work is excluded.
`paper/inciscos2026/inciscos2026.pdf` is the visually reviewed build. The
existing SIPAIM paper subtree remains unchanged.

A standalone Spanish methodology is available in
`reports/methodology/descripcion_metodologica_experimento_eqvae.md`, with its
visually reviewed human-readable PDF in
`output/pdf/descripcion_metodologica_experimento_eqvae.pdf`. It preserves the
scientific intent of the UBC-OCEAN data construction, the matched
ResNet-18-derived VAEs, continuous-`SO(2)` tensor operations, corruption model,
downstream tasks, visualization mathematics and the active Stage A2
continuation, while omitting incidental execution and file-fragment history.

[Spec 0054](docs/specs/0054-cross-validated-mil-dynamics-and-telemetry.md) is a
draft plan for a future classifier rerun on the 361 VAE-development WSI. It
defines paired five-fold OOF learning dynamics, an optional three-fold
sensitivity/inner-selection role, repeated initialization/order trajectories,
tiered optimization and MIL-attention telemetry, and a later full-data refit.
Its counterfactual boundary diagnostics now cover transferable per-step
step-size selection, SAM, orthogonal-gradient and StableMax hypotheses,
per-example/noise regularization, hybrid Muon direction quality and offline
PANTHER prototype suitability without activating any of those methods. A
later, narrow architecture screen now retains mean pooling and a full-data
matched gated-ABMIL control, then admits only mechanism-selected MIL families;
fold-aware nuisance probes, per-WSI gradient sketches and representation
trajectories distinguish memorization from dataset shortcuts. Its immediate
recommended execution is now explicit: an unchanged batch-one Spec 0026/AdamW
five-fold observational anchor on the 361 WSI, with counterfactual diagnostics
unable to affect training, immediately followed within the first campaign by
five-fold effective-batch 4/8 arms under the same exposure-indexed recipe.
Initialization/order repeats use the selected batch; architecture arms remain
later.
The telemetry contract includes a named capture point for every semantic
layer/sublayer, with cheap blockwise forward/gradient/update summaries at T0,
sampled activation-gradient and optimizer statistics at T1, and full
distribution/spectral diagnostics at T2. It also makes runtime packaging
explicit: a Kaggle session is a resumable compute shard, T0/T1 plus a sparse
prebudgeted T2 subset accompanies useful training progress, and the exhaustive
T2 catalogue is distributed across frozen checkpoints rather than repeated at
every boundary. Historical throughput is only a capacity bound; A0 must freeze
the final cadence after measuring each diagnostic family's marginal cost.
The local implementation now provides the non-disruptive observational core:
a compile-compatible T0 reduction wrapper, exact actual-gradient and
actual-AdamW-update summaries, sampled eager T1 layer probes, direct T2-lite
attention/representation reconstruction, resumable learning-dynamics state,
three self-describing cumulative NPZ tables and one atomic `latest.pt` per run
containing RNG, optimizer, scaler, scheduler and exact order/cursor. Focused
local tests cover functional equivalence, compile compatibility,
accepted attention-backend state equivalence, exact first-step gradients/AdamW
state, a real AMP overflow/retry, telemetry resume and exact next-update
continuation. The 361-WSI cohort/fold manifest and compact A0 contract are now
frozen. A thin Kaggle A0 runner audits mounted patch identities and binary
headers, emits the physical-row instance manifest, reconstructs paired
frozen-VAE bags for a class-sentinel and median/P99/maximum cost panel, proves
first-update T0 equivalence, and measures paired T0/T1/T2-lite cost before full
training. The remote A0 has not yet been submitted; cadence and the full
training runner stay pending its measurements. Heavy T2 remains future work.
The separate 152-WSI cohort
remains outside all new decisions and must be reported as historically exposed
rather than a newly sealed test population.

For accepted Stage A1 only, the numerical method is fixed: direct disposable JVP/VJP graphs,
matrix-free `Gv = J^T(Jv)`, eager FP32 decoder products, FP64 Lanczos
tridiagonal solves, finite-difference epsilon `0.008`, JVP microbatch `4`,
Lanczos depth `m=64`, `r=64` independent probes and two-sided 99% confidence
intervals. Stage A2 keeps chart JVP/VJP products eager and compiles only the
repeated frozen-decoder eight-edge energy closure, without numerical or runtime
gates.

The first scientific stage is complete. The corrected next-stage design uses
only exact `C4` states at ranks 0 and 12. It separates literal-endpoint decoder-
energy paths from free-terminal geodesic-consistency rollouts on validated fixed local immersions,
uses distinct searches for approximate decoder-insensitive bridges, decodes
all intermediate path points, tests exact-`C4` ambient decoded covariance, and
measures point return separately from tangent/frame holonomy. Free continuation
uses the first quarter only; its future anchors and closure scorer are isolated
until predictions are computed. `q` sectors are descriptive four-point DFT
diagnostics, not the primary chart or evidence for a continuous action.

The scientific Stage A2 solver now implements the frozen compact machine
contract. It fixes path-local rank handling, the IVP integrator and common
compute budget without scientific pass/fail tolerances. No pooled cross-patch
action is part of Stage A2.

## Accepted scientific evidence

- Both VAEs are frozen at 60,000 updates and use posterior-`mu` latents of
  shape `(16,32,32)`.
- Exact `C4` evaluation covers all 25 fixed validation patches. The prescribed
  `SO(2)` decoder action has image-space RMS error around `1e-6`; the normal
  VAE medians are approximately `0.165--0.190`.
- This decoder result does not imply strict encoder equivariance. Raw
  posterior-`mu` and `logvar` action errors do not favor the `SO(2)` VAE.
- At ranks 0 and 12 over four exact angles, the `SO(2)` pullback-metric trace
  is `0.895--0.959` of the matched normal value, and its estimated
  `d95`/`d99` is lower in all eight matched states.
- The spectral result is two-anchor evidence, not a patch-population estimate.
  It does not establish a continuous-angle action, a geodesic rotation orbit or
  a global quotient structure.

The final Stage A1 kernel is
`maximshtefan/eqvae-fg-stage-a1-c4-28a08ab5/1`. Its authenticated result is
stored at `runs/kaggle/functional_geometry_stage_a1_c4_slq_v1/` with SHA-256
`f59bf4eb7bba136c02cca24b237414da1dca23e4b94b781c4eb3d5978840e449`.

## Live implementation

- `src/eqvae/models/local_global_mil.py` and
  `src/eqvae/models/local_attention_candidates.py`: restored accepted
  Spec 0026 MIL topology and local-attention backend, with an additive
  representation-return path that leaves the ordinary forward unchanged.
- `src/eqvae/training/mil_dynamics.py`: compile-compatible T0 reductions,
  resumable per-WSI dynamics, exact actual-gradient/update diagnostics and
  sampled eager T1 layer/optimizer probes.
- `src/eqvae/training/mil_t2_lite.py`: direct diagnostic reconstruction of
  local radial/null attention, global gates and weights, final CLS attention,
  SwiGLU products, residual ratios and representation spectra.
- `src/eqvae/training/mil_telemetry_io.py` and
  `src/eqvae/training/mil_dynamics_checkpoint.py`: small cumulative NPZ tables
  and direct atomic `latest.pt`/`final.pt` checkpoint/resume, including mandatory
  WSI-dynamics restoration and no object-store/manifest machinery.
- `src/eqvae/training/mil_dynamics_step.py`: public-API AMP step orchestration
  with skipped-attempt rollback semantics and committed-update dynamics.
- `tests/test_spec0054_mil_dynamics.py`: focused mathematical, equivalence,
  persistence and restart tests for the local Spec 0054 primitives.
- `docs/data/spec0054_cohort_folds.csv` and
  `docs/data/spec0054_a0_contract.json`: deterministic five-fold cohort and
  source/hash/CRC-bound A0 contract; fold 0 is the cost-probe context only.
- `experiments/spec0054_mil_a0_probe.py`: bounded real-data A0 audit and paired
  telemetry-equivalence/cost runner. It is not the five-fold training runner.

- `src/eqvae/evaluation/functional_geometry_rla.py`: direct, disposable
  decoder JVP/VJP linearization.
- `src/eqvae/evaluation/functional_geometry_slq.py`: matrix-free pullback
  metric action and deterministic SLQ summaries.
- `docs/data/functional_geometry_stage_a1_contract.json`: compact machine
  contract for the accepted Stage A1 inputs and parameters.
- Stage A1 code is preserved by Git history and immutable Kaggle kernel
  `maximshtefan/eqvae-fg-stage-a1-c4-28a08ab5/1`.
- `docs/data/functional_geometry_stage_a2_contract.json`: frozen scientific
  path, path-local `U_0`/SVD, one-shot shooting-consistency and refinement
  numerics.
- `src/eqvae/evaluation/functional_geometry_stage_a2.py`: deterministic
  path-local chart construction, one-shot RK2 shooting and discrete transport.
- `experiments/spec0053_stage_a2_calibration.py`: the aggregation-only
  scientific runner; it loads the 44 completed paths from exact dataset paths
  and computes the remaining geometry with one frozen model per GPU.
- `kaggle/kernels/functional_geometry_stage_a2/`: explicit thin entrypoint for
  the pending scientific run; `scripts/kaggle_kernel.sh` remains the single
  generic upload path.
- `tests/test_functional_geometry_slq.py`: reusable SLQ/JVP mathematics.
- `tests/test_functional_geometry_stage_a2.py`: focused synthetic chart,
  shooting, second-directional and transport seams.

## Active Kaggle execution

Scientific kernel
`maximusshtefan/eqvae-functional-geometry-stage-a2/1` was submitted at
`2026-09-17T13:58:28-05:00` from public commit
`f0ff9c3225530d1abd24709bad9b17a847d2cd93`, reached `RUNNING`, and was
user-cancelled after about nine minutes once the missing resumability was
identified; Kaggle reports `CANCEL_ACKNOWLEDGED`. It completed no path and
produced no scientific result. Kaggle normalized the initial metadata title to
this URL slug; the local metadata now matches it.

Replacement version
`maximusshtefan/eqvae-functional-geometry-stage-a2/2` was submitted at
`2026-09-17T20:15:36-05:00` from public commit
`1715b0070349ee7555123f7a12534f15c949749c` and ended
`CANCEL_ACKNOWLEDGED` near the session limit without a traceback. Its normal
worker completed all 22 paths; the `SO(2)` worker completed 13 of 22. The 35
atomic checkpoints were downloaded locally; only `rank_12_bridge_3` and the
eight rank-12 `encoded`/`prescribed` sides remained.
Version `maximusshtefan/eqvae-functional-geometry-stage-a2/3` was submitted at
`2026-09-18T09:37:20-05:00` from public commit
`f339209` and ended `CANCEL_ACKNOWLEDGED`. Its two-GPU first phase completed
those nine paths in `14191.29` seconds. The mounted v2 dataset exposed one
`checkpoints.zip`, while the runner searched for unpacked `.pt` files; its
fallback therefore recomputed old paths until the session limit. The local
union of v2 and v3 now contains 44/44 loadable checkpoints. The active runner
has no optimizer, checkpoint writer, discovery, fallback, resume branch, or
shard phase: it directly loads the 44 required flat files from exact paths and
performs only the two-model aggregate geometry. Dataset
`maximusshtefan/eqvae-stage-a2-complete-checkpoints/1` contains those 44 flat
files. Scientific kernel
`maximusshtefan/eqvae-functional-geometry-stage-a2/4` was submitted at
`2026-09-19T01:43:17-05:00` from public commit
`3ddda735457a70cb2444bf98f48a6a9d3f0a3e9a` and failed in 29 seconds before
loading any path. The checkpoint dataset was attached with Kaggle's short
`/kaggle/input/eqvae-stage-a2-complete-checkpoints` mount, while the runner used
the owner-qualified resource-cache path. The correction changes only that exact
root; missing checkpoint files still fail directly in `torch.load`. Replacement
version `maximusshtefan/eqvae-functional-geometry-stage-a2/5` was submitted at
`2026-09-19T02:22:31-05:00` after public commit `b41ebb9`; its thin entrypoint
cloned and executed `dbbe0210a422c4e4b6aafcc55fc71257c5b07b5b`. Both model
workers completed and wrote their JSON and tensor outputs, but the kernel ended
`ERROR` after `8332` seconds because the final convenience aggregator tried to
zip model-specific transport spectra of unequal lengths. The computation was
not repeated. Its outputs are stored under
`runs/kaggle/functional_geometry_stage_a2_v5/`; the corrected local combined
`stage_a2_result.json` has status `complete` and SHA-256
`256965a5f971eae87f63a2285fcf421d9c099edc5b56451c6139f14be92cea86`.

Stage A2b is a post-hoc convergence experiment, not a replacement for the
frozen Stage A2 result. For each model and ranks 0/12 it warm-starts the first
encoded side, first prescribed side, and quarter-turn representative bridge
from their published best full-latent paths. Each receives 1,024 additional
Adam steps with energy-driven LR plateau reduction. One GPU owns each frozen
model. The new runner saves the current and best paths plus optimizer and
scheduler state every 128 steps; it has no path discovery or recomputation
fallback. Because the Stage A2 files did not store Adam state, this first A2b
run is explicitly a warm restart. Its scientific outputs are convergence,
gradient, path-change, and decoded bottleneck measurements.
Version 1 failed before model loading because this new kernel received the
checkpoint dataset at Kaggle's owner-qualified mount while the runner used the
legacy short alias. Version 2 fixed only that exact constant and completed all
12 paths in `34498` seconds. Its outputs are downloaded under
`runs/kaggle/functional_geometry_stage_a2b_v2/`.

All 12 A2b paths reached their lowest energy at additional step 1,024 and kept
the initial LR `.0002209708691`; the plateau scheduler never fired. Relative
energy reductions were `7.00%--16.16%` for normal sides, `5.18%--21.77%` for
`SO(2)` sides, `9.88%--14.81%` for normal bridges, and `44.24%--46.09%` for
`SO(2)` bridges. The `SO(2)` quarter-turn bridge `E/Delta^2` improved from
`77.62` to `43.28` at rank 0 and from `102.34` to `55.17` at rank 12; its
length/endpoint ratio improved from `8.68` to `6.55` and from `9.97` to `7.41`.
Its decoded bottleneck remains `2.91x` and `3.29x` the endpoint gap, so A2b
strengthens the solver-limited diagnosis but still does not establish a
decoder-insensitive connected fiber. The prescribed `SO(2)` side
`E/Delta^2` improved from `1.658` to `1.297` and from `1.502` to `1.258`, so
part of its earlier excess curvature was optimization error.

The 12 exact A2b states are now available in the private Kaggle dataset
`maximusshtefan/eqvae-stage-a2b-continuation-checkpoints`. The continuation
runner directly restores each `current_path`, Adam state and plateau-scheduler
state; a missing state fails at its exact `torch.load` path.
Kernel `maximusshtefan/eqvae-functional-geometry-stage-a2b/3` was submitted
from public commit `4ae6ce8` at `2026-09-20T15:13:51-05:00` and failed before
optimization because the attached dataset used Kaggle's short mount while the
runner used the owner-qualified path. The correction changes only that root.
Version 4 then reached `SUCCESS` in 9h42 (`34924` seconds), completing all 12 exact continuations,
restoring their current paths, Adam states and schedulers at step 1,024 and
retaining checkpoints through step 2,048. Every retained best is at step 2,048
and the LR remains `.0002209708691`. The normal quarter-turn bridges reached
`E/Delta^2=1.117`/`1.077` (ranks 0/12); the `SO(2)` bridges reached
`29.426`/`34.765`, with decoded bottlenecks `2.35x`/`2.54x` their endpoint
gaps. These state files are available privately as
`maximusshtefan/eqvae-stage-a2c-bridge-checkpoints`, containing exactly the
four bridge checkpoints required next.

The active plan is deliberately sequential:

1. Continue only `bridge_1` at ranks 0 and 12 for normal and `SO(2)` from
   their exact step-2,048 `current_path`, Adam and scheduler states for another
   2,048 steps, through cumulative step 4,096. Keep one frozen model per GPU,
   checkpoints every 128 steps, and no recomputation or warm restart.
2. Once energy and gradient histories settle, reoptimize selected converged
   paths at `K=64` and compare decoded curves, not only their energies.
3. Characterize possible fibers with full-latent Jacobian small modes,
   predictor--corrector continuation and decoded bottleneck; do not use the
   horizontal shooting chart `U` for this search.
4. Measure the continuous prescribed `SO(2)` orbit and its speed, covariance
   and geodesic curvature, then compare it with encoded and variational paths.
5. Only after those steps, test matrix-free full-latent shooting, periodic
   multiple shooting with closure in a validated equivalence class, and then
   holonomy. Do not repeat arbitrary `d=32/128/256` chart sweeps.

Private calibration `maximshtefan/eqvae-fg-stage-a2-calibration/3` failed after
18 seconds, before model loading, because the direct-mounted patch path omitted
Kaggle's `/datasets/<owner>/` prefix. The numerical solver and compilation were
never reached. The correction removes the duplicate patch constant and reads
the canonical path already stored in the fixed selector; the weight root is
derived from the contract's owner-qualified locator.

Version `maximshtefan/eqvae-fg-stage-a2-calibration/4` failed because a compiled
gradient tolerance was treated as a fatal gate for the normal VAE. The `SO(2)`
worker nevertheless completed all candidates. The next version removes that
gate, eager/compiled comparisons, runtime selection, memory ceilings and
per-candidate exception wrappers; compilation is now a direct execution choice.

Version `maximshtefan/eqvae-fg-stage-a2-calibration/5` completed successfully
in `3942.73` seconds. Kaggle executed commit
`592b2519213835f5c2eba3eb99fc100a787ec8f4`; both compiled workers completed
all candidates without OOM. The result is stored under
`runs/kaggle/functional_geometry_stage_a2_calibration_v5/`. Its automatic
decision is `unresolved` with the single blocker
`neither_d128_nor_full_latent_met_the_common_budget`. The reason is not a
runtime or conditioning failure: `d=128` has worst minimum singular-value
ratio `.03647`, but its best energies remain `15.05%--29.76%` above the paired
full-latent controls. All full-latent controls reduce energy by
`23.94%--43.27%`; only normal rank 16 encoded touches the arbitrary `.2`
line-deviation trust tube. Several paths are still descending at iteration
128. Do not repeat the reduced-chart arms. The focused follow-up deleted all
`d=32/128` candidates, chart construction, trust projection and automatic
acceptance gates. It ran four paths total: two models by the prescribed-rank-4
and encoded-rank-16 calibration routes.

Replacement-account kernel
`maximusshtefan/eqvae-fg-stage-a2-calibration/1` completed in `5874.02` seconds
from public commit `99d908c75139f88b763433c985ada3e5fe5e9bdd`; both workers
exited normally and all four full-latent `K=32` paths completed 512 steps.
Best energy reductions from the linear initialization are `28.55%`, `36.40%`,
`46.78%` and `23.95%`. Three candidates attain their recorded best at step
512; the `SO(2)` rank-16 encoded path is best at 384 and is `3.94%` worse at
512. The common step 384 has the smallest worst candidate gap (`3.38%`), but
the fixed scientific rule is a common 512-step ceiling with best-iterate
retention rather than a model-specific stop. Maximum line-deviation fractions
are `.208--.283`, confirming that the discarded `.2` trust tube was binding.
Outputs are downloaded under
`runs/kaggle/functional_geometry_stage_a2_full_v1/`.

Private calibration version
`maximshtefan/eqvae-fg-stage-a2-calibration/1` was accepted at
`2026-09-15T04:36:13-05:00`, completed in `2098.65` seconds, and was downloaded
under
`runs/kaggle/functional_geometry_stage_a2_calibration_v1/`. Its immutable
launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-fg-stage-a2-calibration/v0001.json`;
the uploaded `run.py` SHA-256 is
`51d2f1b6acc3e589d439877f700698bab63472745d857571562ab514badbf32d`.

The output receipt and all downloaded SHA-256 values verify. The selection is
`unresolved`, SHA-256
`a17542a87ba73f318505b45d2f73f27fd09fbf4787ca096f11421f5a02d527d1`,
with exact blockers `no_sampled_linearity_radius_passed` and
`K16_or_K32_runtime_probe_failed`. The `.01` radius already has worst sampled
transverse error `.34225`; `K=32` OOMs for `SO(2)` and `K=64` OOMs for both.
Charts are conditioned, FD `.016` passes, and path-coordinate Adam `.005/32`
reduces energy without trust contact in dimensions `16/32`, but those are
partial diagnostics, not a scientific contract. The VAEs remained frozen:
Adam held only interior path coordinates and the final state hash matched.

Do not access final-pilot inputs or implement scientific Stage A2 from either
calibration. Private v2
`maximshtefan/eqvae-fg-stage-a2-calibration/2` completed successfully in
`35906.99` seconds and is downloaded under
`runs/kaggle/functional_geometry_stage_a2_calibration_v2/`. Its decision is
still `unresolved`; the selection SHA-256 is
`4ae2b22f98419fe2ddb3777b14a53a49b834a3e7186c8a776d86c9523e40a4b1`.
Both workers exited normally, all `K=8,16,32,64` runtime probes completed and
peak reserved memory stayed below the fixed ceiling.

V2 selects `d=16/32`, FD epsilon `.016` and Adam LR `.005` only as partial
diagnostics. No sampled radius passes the fixed worst-case `.05` criterion:
radius `.0015` is best with worst error `.061422`; 1190/1200 observations pass
and p99 is `.048893`, with all ten failures confined to prescribed `SO(2)`
midpoint directions. All 32 optimizer candidates improve by at least `5.046%`
without trust contact, but no single recorded milestone is within 1% of every
candidate's own best and six candidates attain their best only at iteration
384. The emitted blocker `optimizer_did_not_improve_within_grid` is therefore
misnamed: improvement occurred; common near-best settling did not. V2 does not
justify the scientific solver/final-pilot run.

V3 tests only the unresolved choices under a 120-minute ceiling: `d=32`,
`d=128` and direct full-latent controls at `K=16`, plus `d=128` and full latent
at `K=32` for the harder prescribed route. Every candidate runs one continuous
128-step Adam trajectory with `.005 sqrt(32/d)`. A reduced chart can be selected
only when its paired full-latent control also improves, settles and avoids the
trust boundary. The full control uses direct `(K-1)x16x32x32` offsets, never a
materialized `16384x16384` identity.

## Scientific boundary

The target is continuous `SO(2)`, although the next pilot deliberately uses
only exact quarter-turns. Similar endpoint decodes do not establish a fiber;
approximate decoder equivalence requires a connected bridge. A small fixed
chart defines only a restricted decoder immersion unless the stronger full-
quotient rank/transversality criteria pass. Geodesic continuation, point closure,
tangent return and frame holonomy are distinct hypotheses. Holonomy is not
torsion. A return to the same accepted decoder-equivalence class at a different
latent representative is recorded as a representative return defect (monodromy
diagnostic), and is called monodromy only if the required fiber/quotient/gauge
structure is validated. Sealed-test results remain unavailable for tuning.

## Verification state

The reduced active suite passes all 23 tests, including the focused Stage A2
chart, shooting, second-directional and transport seams.
Python compilation passes. Ruff now checks only
`E9/F6/F7/F82` runtime-error families and Basedpyright runs in `basic` mode; do
not restore `ALL`, exhaustive annotation/docstring rules or strict tensor typing.
Both checks pass. The Stage A2 thin package validates locally; the private
flat-file checkpoint dataset is uploaded and ready.
Historical calibration kernels and contracts are preserved by Git
and their authenticated Kaggle results; the generic shell script is the only
upload path. Personal
references remain ignored and no live code depends on the local experiment
archive.
