# Current Repository Status

Last updated: 2026-09-15

## Active frontier

[Spec 0053](docs/specs/0053-functional-riemannian-latent-geometry.md) is the
single live contract for comparing the functional latent geometry of the frozen
normal and continuous-`SO(2)` VAEs.

The accepted numerical method is fixed: direct disposable JVP/VJP graphs,
matrix-free `Gv = J^T(Jv)`, eager FP32 decoder products, FP64 Lanczos
tridiagonal solves, finite-difference epsilon `0.008`, JVP microbatch `4`,
Lanczos depth `m=64`, `r=64` independent probes and two-sided 99% confidence
intervals. Compilation remains disabled for this analysis.

The first scientific stage is complete. The next bounded experiment uses only
the exact `C4` states at ranks 0 and 12. It will compare the encoded and
prescribed-action cycles, search for decoder-fiber bridges, solve all four
adjacent pullback-metric geodesics, test tangent transport and action covariance,
and perform a free continuation in which only `z_0,z_1` may determine
`z_2,z_3,z_0`. Machine tolerances, the solver budget, focused tests and the
Kaggle package remain to be implemented.

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
- `tests/test_functional_geometry_slq.py` and
  `tests/test_stage_a1_c4_slq_kernel.py`: focused numerical and package tests.

## Scientific boundary

The target is continuous `SO(2)`, although the next pilot deliberately uses
only exact quarter-turns. Decoder-fiber equivalence means agreement in the
decoder-induced quotient, not equality of latent tensors. Parallel transport
and geodesic closure are hypotheses to test, not assumptions. Sealed-test
results remain unavailable for tuning.

## Verification state

The 10 focused SLQ/kernel tests pass. Focused Ruff and format checks, Python
compilation, shell syntax, embedded-payload build/verification, kernel
`validate`/`check` and `git diff --check` pass. The generated
`run.py` was removed after verification because the template and compact
payload builder are the source of truth. Preserve unrelated dirty-worktree
changes.
