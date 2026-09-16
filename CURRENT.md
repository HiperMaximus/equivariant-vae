# Current Repository Status

Last updated: 2026-09-16

## Active frontier

[Spec 0053](docs/specs/0053-functional-riemannian-latent-geometry.md) is the
single live contract for comparing the functional latent geometry of the frozen
normal and continuous-`SO(2)` VAEs. Stage A1 is accepted; Stage A2 is
`draft active`. Calibration v1 and v2 completed `unresolved`; the focused v3
solver calibration is running on Kaggle. No calibration result yet justifies
the scientific solver/final-pilot run.

For accepted Stage A1 only, the numerical method is fixed: direct disposable JVP/VJP graphs,
matrix-free `Gv = J^T(Jv)`, eager FP32 decoder products, FP64 Lanczos
tridiagonal solves, finite-difference epsilon `0.008`, JVP microbatch `4`,
Lanczos depth `m=64`, `r=64` independent probes and two-sided 99% confidence
intervals. Stage A2 v3 keeps chart JVP/VJP products eager and may compile only
the repeated frozen-decoder eight-edge energy closure after real-block output,
latent-gradient, memory and settled-runtime validation.

The first scientific stage is complete. The corrected next-stage design uses
only exact `C4` states at ranks 0 and 12. It separates literal-endpoint decoder-
energy paths from free-terminal geodesics on validated fixed local immersions,
uses distinct searches for approximate decoder-insensitive bridges, decodes
all intermediate path points, tests exact-`C4` differential covariance, and
measures point return separately from tangent/frame holonomy. Free continuation
uses the first quarter only; its future anchors and closure scorer are isolated
until predictions are computed. `q` sectors are descriptive four-point DFT
diagnostics, not the primary chart or evidence for a continuous action.

Before the scientific Stage A2 solver may run, the calibration must return a
viable common numerical envelope and a compact machine contract must then fix
the remaining rank/immersion criteria, IVP integrator, scientific tolerances,
and common compute budget. No pooled cross-patch
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
- `src/eqvae/evaluation/functional_geometry_calibration.py`: reduced-chart,
  thin-SVD, path-energy and trust-tube helpers for numerical calibration.
- `docs/data/functional_geometry_stage_a2_calibration_contract.json`: fixed
  non-scientific grid, selection criteria and resource ceilings.
- `experiments/spec0053_stage_a2_calibration.py`: scientific calibration
  orchestration and one worst-case common reducer.
- `kaggle/kernels/functional_geometry_stage_a2_calibration/main.py`: thin
  Kaggle entrypoint; it mounts the published frozen-weight/patch datasets and
  sparse-clones the exact pinned public source commit instead of embedding it.
- `tests/test_functional_geometry_slq.py`: reusable SLQ/JVP mathematics.
- `tests/test_functional_geometry_calibration.py` and
  `tests/test_stage_a2_calibration_kernel.py`: focused helper, leakage,
  reducer and package tests for the calibration.

## Active Kaggle execution

Private calibration `maximshtefan/eqvae-fg-stage-a2-calibration/3` was launched
at `2026-09-16T13:45:38-05:00` and confirmed `RUNNING`. Its uploaded entrypoint
is commit `f2021eed207fcd4358c0d2a38f2abc60e58c1380`; the experiment itself is
pinned to source commit `b9879ecb38f5d25cb182ea4868ddce5899ec7241`.
Inspect after the 120-minute experiment ceiling, around
`2026-09-16T15:45:38-05:00`; do not launch another version while v3 is active.

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
materialized `16384x16384` identity. The machine contract SHA-256 is
`99e09f326be2dcdf787bfc168538e09753dc3275fbde672c204550a59be204a7`.

## Scientific boundary

The target is continuous `SO(2)`, although the next pilot deliberately uses
only exact quarter-turns. Similar endpoint decodes do not establish a fiber;
approximate decoder equivalence requires a connected bridge. A small fixed
chart defines only a restricted decoder immersion unless the stronger full-
quotient rank/transversality criteria pass. Geodesic continuation, point closure,
tangent return and frame holonomy are distinct hypotheses. Holonomy is not
torsion. Sealed-test results remain unavailable for tuning.

## Verification state

The reduced active suite passes all 30 tests, including
chunked-versus-monolithic energy/gradient equality, dimension-normalized
full-latent Adam, valid full-control pairing and frozen-`SO(2)` kernel caching.
Python compilation and direct-kernel build/validate pass. Ruff now checks only
`E9/F6/F7/F82` runtime-error families and Basedpyright runs in `basic` mode; do
not restore `ALL`, exhaustive annotation/docstring rules or strict tensor typing.
Both checks pass. The exact v3 contract SHA-256 is
`99e09f326be2dcdf787bfc168538e09753dc3275fbde672c204550a59be204a7`.
Only the thin Stage A2 Kaggle package remains live. This cleanup deletes 265
versioned files and more than 164,000 lines: historical embedded kernels,
per-campaign builders, runtime/readiness gates, CLIs, consumed training/data/
inference/evaluation machinery, old notebook captures and completed-campaign
tests, plus unused IEEE template samples and manuals. `src/eqvae` is now 14
live modules and 3,157 lines: the two frozen VAE definitions plus active
geometry only. Specs, contracts, hashes, remote locators and accepted results
remain as provenance. The A2 sparse clone reads the byte-identical tracked
fixed-25 selector from `configs/spec0001` instead of the ignored `runs/` tree.
The v1 runner was recovered from its authenticated Kaggle source; the v2
runner is a net 51-line change (143 insertions/92 deletions) rather than the
discarded 2,719-line replacement. Stage A2 has no embedded payload, generated
`run.py`, per-kernel builder, launch-receipt framework or polling controller.
The generic shell runner uploads only metadata plus the direct code file.
Future parameter reruns edit the contract, push the repository and reuse the
same small Kaggle entrypoint. Personal references remain ignored and
outside the repository contract. The ignored local experiment archive was
reduced from about 21 GiB to 1.8 GiB: scratch trees, aborted probes, generated
payloads, duplicate resume weights, consumed probe checkpoints and
non-contract intermediate tensors were removed. All 118 hash-bound
professor-report inputs, informative negative summaries, Stage A1/A2 evidence
and the normal/`SO(2)` frozen checkpoints required by Spec 0053 remain and
verify. Fresh clones contain none of this ignored archive, and no live code
depends on it.
