# Current Repository Status

Last updated: 2026-09-19

## Active frontier

[Spec 0053](docs/specs/0053-functional-riemannian-latent-geometry.md) is the
single live contract for comparing the functional latent geometry of the frozen
normal and continuous-`SO(2)` VAEs. Stage A1 is accepted; Stage A2 is
`draft active`. Calibration v1 and v2 completed `unresolved`; Kaggle v3 failed
during input startup and v4 on an unnecessary compile gate. Simplified v5
completed successfully and resolved the search-space question: `d=128` is
conditioned but materially restricts the optimized paths relative to direct
full-latent control. The replacement-account full-latent calibration completed
successfully and closes the numerical search: use `K=32`, at most 512 Adam
steps and retain the lowest-energy iterate. The compact Stage A2 numerical
contract is now frozen: it constructs `U` deterministically from the first
full-latent side, uses one common FP32-aware SVD convention, and specifies a
one-shot shooting consistency IVP plus fixed time and frozen-path quadrature
refinements. The scientific runner and its explicit thin Kaggle entrypoint are
implemented locally; no Stage A2 scientific result exists until that run
completes.
No further hyperparameter calibration, longer-step probe, or plateau-scheduler
rerun is justified. Stage A2 will report scientific residuals continuously for
both models rather than use comparison tolerances as acceptance gates or
runtime errors; only numerical singularity limits which intrinsic quantities
are defined.
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
`2026-09-19T02:22:31-05:00` from public commit
`b41ebb9467221757cb8d88fec0a08a999dc3faef` and is `RUNNING`.

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
