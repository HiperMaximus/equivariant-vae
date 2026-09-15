# Current Repository Status

Last updated: 2026-09-15

## Active frontier

[Spec 0053](docs/specs/0053-functional-riemannian-latent-geometry.md) is the
single live contract for comparing the functional latent geometry of the frozen
normal and continuous-`SO(2)` VAEs. Stage A1 is accepted; Stage A2 is
`draft active`. Its separate pre-scientific numerical calibration probe is
implemented and completed on Kaggle with decision `unresolved`; the calibration
does not yet justify the scientific solver/final-pilot run.

For accepted Stage A1 only, the numerical method is fixed: direct disposable JVP/VJP graphs,
matrix-free `Gv = J^T(Jv)`, eager FP32 decoder products, FP64 Lanczos
tridiagonal solves, finite-difference epsilon `0.008`, JVP microbatch `4`,
Lanczos depth `m=64`, `r=64` independent probes and two-sided 99% confidence
intervals. The new calibration independently tests the Stage A2 epsilon and
other solver parameters. Compilation remains disabled for this analysis.

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
- `kaggle/kernels/functional_geometry_stage_a1_c4_slq/`: final reproducibility
  package for Stage A1.
- `src/eqvae/evaluation/functional_geometry_calibration.py`: reduced-chart,
  thin-SVD, path-energy and trust-tube helpers for numerical calibration.
- `docs/data/functional_geometry_stage_a2_calibration_contract.json`: fixed
  non-scientific grid, selection criteria and resource ceilings.
- `kaggle/kernels/functional_geometry_stage_a2_calibration/`: private dual-T4
  probe with allowlisted patch-byte reads and one worst-case common reducer.
- `tests/test_functional_geometry_slq.py`: reusable SLQ/JVP mathematics.
- `tests/test_functional_geometry_calibration.py` and
  `tests/test_stage_a2_calibration_kernel.py`: focused helper, leakage,
  reducer and package tests for the calibration.

## Active Kaggle execution

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

Do not access final-pilot inputs or implement scientific Stage A2 from v1.
Calibration v2 is implemented and running on Kaggle.
It is a small patch of the recovered v1 runner: the same allowed ranks/WSIs,
frozen checkpoints and common reducer; smaller first-order radii; and exact
eight-edge energy/backward chunks. V1-supported `d=16,32` and Adam LR `.005`
are tested at `K=16,32` in one continuous trajectory per candidate through
checkpoints `0,8,16,32,48,64,96,128,160,192,256,320,384`. There is no
stationary-gradient, rollback, monotonicity or restart condition. If convergence is
first reached only at 384, the ceiling remains unresolved.

Private v2 `maximshtefan/eqvae-fg-stage-a2-calibration/2` was accepted at
`2026-09-15 11:13 -05` and was confirmed `RUNNING`. Its launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-fg-stage-a2-calibration/v0002.json`;
the uploaded `run.py` SHA-256 is
`54784d4e8251ce9d06a60304566a070a31055e2ad2b4e2e1c4e2d1e9ef1e3861`.
Check status around `2026-09-15 18:15 -05`; the configured 690-minute ceiling
ends around `22:43 -05`. No v2 scientific result exists yet.

## Scientific boundary

The target is continuous `SO(2)`, although the next pilot deliberately uses
only exact quarter-turns. Similar endpoint decodes do not establish a fiber;
approximate decoder equivalence requires a connected bridge. A small fixed
chart defines only a restricted decoder immersion unless the stronger full-
quotient rank/transversality criteria pass. Geodesic continuation, point closure,
tangent return and frame holonomy are distinct hypotheses. Holonomy is not
torsion. Sealed-test results remain unavailable for tuning.

## Verification state

The reduced active suite passes all 25 tests, including
chunked-versus-monolithic energy and coordinate-gradient equality. Python
compilation, kernel build/validate/check and authenticated API source checks
pass. The exact contract SHA-256 is
`12f46782d70ebc6d1bf89a75925928753e62bd8aaf2ac6558976185b27bcc93b`.
The v1 runner was recovered from its authenticated Kaggle source; the v2
runner is a net 51-line change (143 insertions/92 deletions) rather than the
discarded 2,719-line replacement. The obsolete preflight/quality/spec-process
layer and 92 historical training, launcher and completed-campaign test files
were removed. Future parameter reruns edit the existing contract/runner and use
only focused correctness checks. Personal `reference/` and the FSQ reference
remain on disk but are ignored and untracked; the obsolete selector that
depended on them was removed.
