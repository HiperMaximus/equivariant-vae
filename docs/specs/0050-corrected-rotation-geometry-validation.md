# Spec 0050: Corrected Rotation Geometry Validation

Status: implemented / accepted
Implementation readiness: complete; accepted private v1 and reviewed professor report
Owner/workstream: fixed-validation post-hoc rotation geometry audit
Last updated: 2026-09-08

## Purpose

Correct the mixed-sign dense-rotation operator, invalidate rather than hide the
affected evidence, and test whether the two frozen VAEs organize input rotation
through a transferable low-dimensional action and an approximate local
content--pose factorization. This is validation-only exploratory evidence on the
existing SHA-pinned fixed 25. Negative results are accepted.

## Non-Goals And Hard Boundaries

- No VAE retraining, checkpoint mutation, tuning, checkpoint selection, or
  architecture change.
- No sealed-test access and no change to sealed reconstruction, WSI diagnosis,
  tissue, or bootstrap conclusions.
- No paper, thesis, Overleaf, GitHub issue, commit, push, or public-data update.
- No global `M = SO(2) x R^n`, disentanglement, bundle-topology, or gauge-theory
  claim.
- No reuse of the consumed Spec 0038 launch authority. This spec owns one new
  guarded inference-only Kaggle authority.
- No result-dependent example selection. Display fixed ranks 0 and 12.

## Blocking Defect And Supersession

`rotation_orbits.continuous_rotate` currently splices exact positive
`torch.rot90` at multiples of 90 degrees into a non-cardinal `affine_grid`
trajectory with the opposite visual orientation. On fixed rank 12 the defect
reproduces exactly:

- input RMS at 87->88 / 88->89 / 89->90 degrees:
  `0.101921 / 0.103464 / 0.240397`;
- input RMS at 90->91 / 91->92 degrees: `0.239876 / 0.102545`;
- 89.999 degrees versus `rot90(+1)` / `rot90(-1)`:
  `0.237478 / 0.000162`.

The old dense run `maximshtefan/eqvae-fixed25-dense-rotation-population/1` is
superseded for scientific use. Its bytes and receipt remain provenance; the new
pipeline must use a distinct output directory and must fail if configured to
write any legacy path. Exact `torch.rot90` results at 90/180/270 degrees remain
valid and must be reported beside corrected evidence. The non-cardinal helper
in `tests/test_so2_vae.py` is also sign-affected; its cardinal values remain
valid and any non-cardinal pinned values must be re-derived after correction.

The affected-artifact inventory is locked as follows. "Superseded" means that
the bytes are preserved but may not support a live scientific claim.

| Artifact or consumer | Producer | Status / required action |
| --- | --- | --- |
| `runs/local/frozen_vae_rotation_orbits/01-latent-orbits.png` | selected-patch dense orbit renderer | Superseded; preserve bytes. |
| `runs/local/frozen_vae_rotation_orbits/02-f1-phase.png` | mixed-sign continuous F1 sweep | Superseded; preserve bytes. |
| `runs/local/frozen_vae_rotation_orbits/07-all25-latent-orbits.{json,png}` | Spec 0038 population kernel/renderer | Superseded; preserve bytes and receipt. |
| `runs/local/frozen_vae_rotation_orbits/rotation-orbits.html` | legacy dense-orbit renderer | Superseded; preserve bytes. |
| `runs/local/frozen_vae_rotation_orbits/manifest.json` | legacy package builder | Historical provenance only; never mutate. |
| `runs/local/frozen_vae_rotation_orbits/03-kernel-mechanism.png` through `06-pointwise-rgb-probe.png` | mechanism and spatial probes not using the mixed dense path | Not invalidated by this sign defect. |
| `README.md`, `docs/repo_goal_and_requirements.md`, `docs/equivariant_vae_transition_plan.md`, `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`, `docs/specs/README.md`, `docs/issue_image_inventory.md` | live repository prose | Replace the 25/25 claim with an explicit provisional/superseded notice before implementation. |
| `scripts/build_professor_final_report.py` and current report Figures 7, 8a, 8b | report builder / Spec 0048 | Stale consumer; replace only after corrected results pass review. Until then, the current report is not valid evidence for dense rotation. |
| `src/eqvae/cli/render_frozen_vae_rotation_orbits.py` | legacy package writer whose default is the preserved directory | Quarantine before correction: refuse the legacy directory and every existing package, so corrected code cannot rewrite old provenance. |
| Existing GitHub attachment/comment | prior authorized external publication | Externally stale; record locally, but do not mutate without separate authorization. |

The stale-consumer gate has two phases. Before launch, live prose must contain
only supersession notices, the legacy writer must be quarantined, and the
report builder/current report are allowlisted as non-runnable stale consumers
awaiting corrected results. After result acceptance and report rebuild, no
retired value, claim or legacy dense input may remain in executable report
code; matches are then allowed only in explicit provenance text and
byte-preserved legacy files. Separate tests enforce each phase.

## Inputs And Fixed Data Contract

| Item | Locked value |
| --- | --- |
| Normal checkpoint | `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`; SHA-256 `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075` |
| SO(2) checkpoint | `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`; SHA-256 `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7` |
| Immutable state bundle | `spec0045_vae_test_input.json` SHA-256 `b5a32ebffd0d88a88d6f21b64ba5c9a23016f05d7a2db0546e442f12a0acecc1`; normal/SO(2) state-file SHA-256 `30064fa414f21deeb7ec5f312467ad4e8a3a93f6e7199e7be77f6ed3b28887c7` / `06802ceb6ba4fb0f46d2b88f751a33a355600db84b412806b563a6d757cb3c12`; state-dict SHA-256 `feb36bc81dabe49352b0e04c8ddfa11f800195a3809edcd005ebd86342a39803` / `62887fc4c9549349fcceed450f16bae381fe1c4f83883652fee425203c197573` |
| Patches | `docs/data/fixed25/originals.pt`; exactly 25 ordered validation patches and their existing unique SHA-256 values |
| Input | FP32 `3x256x256`, reconstructed as `uint8 / 255 * 2 - 1` |
| Posterior | raw FP32 `mu` with shape `16x32x32`; no posterior sampling |
| Internal SO(2) field | final encoder D before `mu_head`: `48F0 + 48F1 = 144x32x32` |
| Angles | integer degrees `0,...,359`; 360 is display-only exact copy of 0 |
| Latent mask | centered radius-14 disk, 616 cells |
| Input masks | full image and centered radius-112 disk |
| Machine contract | `docs/data/spec0050_rotation_geometry_contract.json`; SHA-256 `1dc6975979b120d85680526a3177e30caf93ca2eba7889a4452f2e78ca399756` |

All hashes, source locators, runtime identity, code/payload identity, split, and
output hashes are written to the accepted machine-readable result.

## Mathematical Design Note

1. **Total space, base, fiber, and action.** The observed total space is the
   represented validation subset of `M = E(X)`; a candidate base identifies
   embeddings that differ only by input rotation; a generic fiber is one orbit
   `O_p = {E(R_theta x)}`. Spatial coordinate rotation is architecturally
   prescribed for the SO(2) model's final F0 posterior and is only a
   prespecified comparison action for the normal model. Stabilizers may reduce
   a fiber to `SO(2)/H_p`.
2. **Prescribed versus estimated action.** Architecture prescribes spatial
   rotation for SO(2) final F0 fields and simultaneous spatial plus internal
   vector rotation for SO(2) F1 fields. The reduced generator `A_d` is an
   empirical, basis-dependent approximation fitted in a shared PCA basis; it
   is not the architectural generator.
3. **Shared action versus unrelated curves.** One `A_d` must be fitted using
   only fit patches and 5-degree anchors, then roll out each held-out orbit from
   its single 0-degree state without re-anchoring. Local 1--4 degree
   interpolation from the preceding anchor is secondary and cannot establish a
   shared action. Transfer must beat identity and shuffled controls, explain a
   non-negligible share of raw orbit variance, and remain stable under
   leave-one-fit-patch-out refits.
4. **Equivariance versus factorization.** Small
   `||E(R_theta x)-rho(theta)E(x)||` verifies the action. Factorization further
   requires inverse-action canonicalization to reduce within-patch variation
   relative to between-patch variation while retaining patch identity/content
   distances and suppressing angle leakage.
5. **Local phase and gauge.** An `m=1` plane or nonzero F1 vector defines a
   local phase. Adding a patch-dependent constant phase changes the local
   reference section without changing the represented rotation. Phase is not
   defined at zero amplitude.
6. **Possible stabilizers.** Weakly oriented, near-isotropic, radial, repeated,
   or approximately 180-degree-symmetric patches may have ambiguous phase or a
   `C2`-like stabilizer. They remain in every population result.
7. **Basis invariants.** Explained variance, singular values, subspace
   principal angles, skew-generator eigenfrequencies, held-out prediction,
   harmonic power aggregated over a subspace, and canonicalization performance
   are invariant to orthogonal basis changes. Orthogonality and group-law
   residuals are construction/numerical checks, not learned evidence.
   Individual PCA axes, signs, `A` entries, and patch-local phase origins are not.
8. **Interpretation-changing outcomes.** Loss of the corrected 25/25 result
   retracts the old smoothness claim. Equal generator transfer gives no
   SO(2)-specific advantage. Patch-only transfer rejects a common action. Clean
   F1 with poor `mu` implies the posterior head loses/redistributes explicit
   structure. Failed canonicalization rejects factorization. Phase failures
   concentrated in symmetric patches support only a local stabilizer reading.
9. **Impossible conclusions.** Twenty-five fixed validation patches cannot
   establish global topology, a global section, a global product, population
   generalization, causal architectural attribution, or training-seed
   robustness.
10. **Independent rotation verification.** The corrected affine operator is
    checked against hand-indexed asymmetric impulses and an arrow tensor whose
    expected pixel coordinates are written explicitly, against limits from both
    sides of every cardinal angle, and only then against `torch.rot90(+k)`.
    This prevents two implementations with the same matrix-sign error from
    validating each other.

## Preregistered Hypotheses And Decision Language

All thresholds below are fixed descriptive margins for these checkpoints and
patches. They are not p-values or estimates of a source population.

| Hypothesis | Primary fixed-sample diagnostic | Support rule |
| --- | --- | --- |
| H0 continuity-only null | All corrected curves and input paths | Retained unless a stronger rule below passes; closure is ignored. |
| H1 rotational regularity | Input-normalized raw-`mu` local-linearity ratio | `SO(2)` median at least 10% lower than normal and favorable in at least 18/25 pairs. Step CV, curvature, exclusions and subsampling are required corroborating/contradicting secondary results. |
| H2 harmonic organization | Raw-orbit `m=1,...,6` power fraction and effective-frequency count | `SO(2)` must have at least 10% higher low-harmonic power fraction and at least 10% lower effective-frequency count, each favorable in at least 18/25 pairs; `m=1` power is reported separately, not required when morphology carries legitimate higher harmonics. |
| H3 internal F1 transformation | Per-copy patch-level medians | A copy is clean only if valid on at least 18/25 patches, `m=1` purity `>=0.75`, phase slope within `0.10` of `+1`, wrapped phase RMSE `<=0.35` rad, amplitude CV `<=0.25`, and full-field residual `<=0.25`. "Substantial fraction" means at least 24/48 clean copies. |
| H4 shared effective action | Strict held-out one-view lifted prediction | The complete rule is in Experiment 6. Coarse-orbit-conditioned or locally re-anchored prediction cannot pass H4. |
| H5 group composition | Numerical residual of `exp(theta A)` and independently fitted Procrustes powers | Only a numerical consistency check; it cannot independently support H5 because the exponential law is imposed. Empirical support comes from H4 transfer across angular horizons. |
| H6 local content--pose factorization | Within reduction plus between-content/retrieval/distance preservation and angle leakage | The complete conjunctive rule is in Experiment 7. Partial passage is unresolved. |
| H7 bundle/stabilizer compatibility | Valid isolated/repeated representation plane and patch-level association with self-symmetry | Only the restricted rule in Experiment 8 can support compatibility; it never establishes topology or a global product. |

For the architecturally prescribed final-`mu` action, "empirically verified"
requires a median per-patch full-field normalized residual `<=0.25` and at least
20/25 patch medians `<=0.25`; `>0.50` is called poor and intermediate values
are unresolved. Thus "clean F1 with poor mu" and its converse are operational,
not chosen after seeing results. Every primary table includes all patch values,
the favorable count and the stated margin; secondary endpoints remain
exploratory.

## Stage 0 Rotation Audit And Correct Operator

Positive rotation is the existing exact-quarter convention
`torch.rot90(values, +k, (-2,-1))`. `grid_sample` consumes output-to-input
coordinates; therefore the corrected sampling matrix for a positive displayed
angle is `[[cos,-sin,0],[sin,cos,0]]` in normalized `(x,y)` coordinates. The
dense operator always builds one explicit normalized pixel-center affine grid
and uses one bilinear `grid_sample` path, including cardinal angles. The
explicit grid avoids `affine_grid`'s observed FP32 cardinal-coordinate drift on
one-pixel fixtures. Cardinal sine/cosine values are snapped to exact `0,+/-1`,
but no `rot90` branch is spliced into the dense trajectory. Exact
`torch.rot90` remains a separate control.

Regression tests use asymmetric impulses/arrow arms with hand-declared target
coordinates. Required checks:

- identity, correct positive handedness, and limits from both sides at 90, 180,
  and 270 degrees;
- inverse `R_theta R_-theta x ~= x` and composition
  `R_alpha R_beta x ~= R_(alpha+beta) x`, reported rather than treated as exact;
- errors on both full images and centered disks;
- raw-input one-degree step RMS, step CV, local-linearity, path length,
  tangent/curvature distributions, and cardinal-neighborhood diagnostics;
- the same quantities after excluding cyclic steps within +/-2 degrees of
  `0,90,180,270` and at 1-, 2-, and 5-degree subsampling.

The independent fixture contract covers both `32x32` and `256x256` even grids.
It declares source/target pixels for exact quarter turns and uses an analytic
off-axis Gaussian/arrow first moment for 17, 31, and 73 degrees; the observed
orientation must be within 0.5 degrees and 0.02 normalized-coordinate units of
the hand-computed target. Uniform-grid cardinal output must agree with exact
`rot90` to maximum absolute error `2e-6`; the RMS at `k*90 +/- 0.001` degrees
must be below `5e-4` from the uniform-grid cardinal output on both full and
centered-disk fixtures. Fixture normalized RMS inverse and composition errors
must each be below `0.03`; fixed-25 errors are recorded without pretending that
interpolation is exact. Explicit mutant tests reverse the sign, swap F1
components, confuse degrees/radians, reverse matrix multiplication, mismatch
`align_corners` between grid construction and sampling, and change padding;
each must fail at least one hand-derived fixture. Exact fixtures are
deterministic FP32 tensors: an interior off-axis Gaussian (`sigma=0.2` in
normalized units) plus one-pixel-width L-arrow for orientation and a separate
boundary-touching constant wedge for
padding. Their normalized-coordinate formula, amplitude and support are encoded
once in the immutable machine contract and used unchanged on CPU and CUDA;
device-specific measured errors are recorded. A consistent same-size
`align_corners=True` transform is not treated as a faulty mutant because it can
be geometrically equivalent.

The packed pre-head layout is locked as channels `[48 F0 | 48 F1 copies]`,
where F1 copy `j` occupies packed channels `48+2j` (x/cos component) and
`49+2j` (y/sin component). Its candidate action is operationally
`T_theta h(p)=rho_1(theta) h(R_-theta p)` with
`rho_1=[[cos,-sin],[sin,cos]]`. An independent packed-field fixture declares
both the component and pixel targets of a single off-center F1 impulse at +90
degrees and at one non-cardinal angle. It must reject `rho_1(-theta)`, component
swaps, and spatial-only rotation, then cross-check the repository
`representation_matrix`/`escnn` oracle only after the hand-derived test passes.

Input and latent RMS use root-mean-square per scalar, so dimensionality alone
does not inflate them. Input-normalized latent results include
`L_latent/(L_input+eps)`, latent/input path-length ratio, and the CV of the
per-angle speed gain `step_latent/(step_input+eps)`. Both centered-disk and full
input controls are retained. Epsilon is `1e-8` in the recorded FP32 metric
contract.

## Experiment 1 Corrected Population Orbits

For every patch/model, persist metrics from disk-masked raw `mu`, not PCA:

- cyclic first and second differences, local-linearity ratio, step-size CV,
  path length, tangent norm, central-difference acceleration, geometric
  curvature, and their angle distributions;
- 1/2/5-degree robustness and +/-2-degree cardinal exclusion;
- two-PC and six-PC cumulative explained variance;
- paired SO(2)-minus-normal values, medians, favorable sign counts, and
  percentile intervals for the median paired difference from 10,000 paired
  patch resamples using seed `3573134496`.

Intervals are descriptive for this fixed set, not population confidence
intervals or significance tests. Closure is never scored because 360 is copied
from 0 for display only.

## Experiment 2 Local PCA Geometry

For fixed ranks 0 and 12, fit one PCA per patch/model over all 360 disk-masked
raw posterior means. Preserve scores PC1--PC6, component and cumulative
explained variance, cardinal markers, angle colors, equal data units per pixel,
PC1--PC2, PC1--PC3, PC2--PC3, a fixed orthographic 3D PC1--PC3 view, and all 15
pairwise PC1--PC6 planes. Coordinates are patch-local; axes, signs, and scales
are not compared between panels.

## Experiment 3 Local Rotation Tangents

Use cyclic centered differences with radians:

`v(theta) = [z(theta+delta)-z(theta-delta)]/(2 delta)`.

Report tangent norm, consecutive tangent cosine/turning angle, acceleration,
geometric curvature, top singular values of the full tangent trajectory,
90/95/99% tangent-span dimensions, and rank-one explained variance in the fixed
five-angle window `{-2,-1,0,1,2}` around zero. For each patch define
`T_i=[v_i(theta_1),...,v_i(theta_360)] in R^(D x 360)`; patch-local rotational
subspaces are the first two left singular vectors of `T_i`. Summarize all
pairwise principal angles within each model. Do not compare absolute latent
directions between separately trained models.

## Experiment 4 Harmonic Organization

Apply the length-360 real FFT to every centered PC1--PC6 score. Real harmonic
power doubles non-DC/non-Nyquist positive-frequency energy and is normalized by
power over all frequencies, not merely `m=0,...,6`. Persist/display individual
PC heatmaps only as coordinates, then report basis-invariant multivariate power
`H_m=2||hat a(m)||_2^2`, spectral entropy/effective-frequency count, and the
singular values of `[Re(hat a(1)), Im(hat a(1))] in R^(6 x 2)`. Report each
harmonic both conditional on PC1--PC6 and relative to total raw-orbit energy,
together with six-PC capture and relevant eigengaps. Higher harmonics are
described, not automatically penalized.

## Experiment 5 All 48 Internal F1 Copies

For each patch, angle, and copy, retain the disk-pooled complex vector and the
full-field normalized residual after spatial rotation plus internal
`[[cos,-sin],[sin,cos]]` rotation. Full activations are streamed and not saved.
For every copy retain its descriptive rows, but aggregate scientific summaries
first within patch and then across the 25 patches:

- `|m|=1` harmonic purity;
- unwrapped phase slope, circular phase error, amplitude CV;
- full-field equivariance residual;
- fraction of angles with pooled magnitude below `1e-6`;
- valid patch/copy count, where the initial pooled magnitude is at least
  `1e-6` and at least 90% of angles remain above it.

Disk pooling is never the sole failure criterion. Exact-quarter F1 residuals
are retained as a separate interpolation-free control.

## Experiment 6 Shared Low-Dimensional Action

Patch assignment is fixed before corrected outputs by sorting
`SHA256("spec0050.shared-action.split.v1:" + patch_sha256)`:

- fit ranks: `23,22,1,8,4,24,9,20,11,10,0,14,5,16,7,17,12`;
- held-out ranks: `18,2,21,19,3,13,15,6`.

Angles divisible by 5 are fit anchors; the other integer angles are held out.
For fit patches only, let `c_i` be the mean of their 72 anchors and let `g` be
the mean across every fit-patch anchor. A deterministic randomized range finder
with seed `4158411771`, fixed oversampling 6, and two power iterations fits one
shared rank-6 PCA basis `U` on fit-patch residuals `z_i(theta)-c_i`; dimensions
`d=1,...,6` use its nested leading columns.

For every `d`, report shared-basis variance captured on fit and held-out patches
relative to their full raw centered-orbit energy. Record the generator design
matrix singular values, numerical rank, condition number, and excited energy
per invariant plane. Fit two skew-symmetric generators by explicit least
squares over the `d(d-1)/2` independent parameters and central 5-degree
derivatives: (a) the requested centered conditional generator on
`U_d^T[z_i(theta)-c_i]`; and (b) the strict global generator on
`U_d^T[z_i(theta)-g]`. Cross-check each against determinant-`+1` orthogonal
Procrustes on 5-degree steps.

Only the strict generator can support H4. For each held-out patch it predicts
every withheld angle using the global fit-only center and the patch's 0-degree
embedding alone:
`z_hat(theta)=g + U_d exp(theta A_d)U_d^T[z_i(0)-g]
+ (I-U_dU_d^T)[z_i(0)-g]`.
No held-out-patch orbit center or other view is available to this prediction.
The conditional generator may separately use a held-out `c_i` estimated from
its 72 anchors; it is labeled coarse-orbit-conditioned dynamics and cannot pass
H4. Preceding-5-degree-anchor predictions are a third, local interpolation
diagnostic. Horizon plots expose drift for all three modes.

For patch `i`, lifted NRMSE is
`sqrt(mean_withheld,D (z_hat-z)^2) /
(sqrt(mean_withheld,D (z-mean_withheld z)^2)+eps)` and lifted
`R2=1-SSE/SST` uses the same held-out mean and scalars. Projected versions use
the identical formulas in `d` dimensions. Compute these per patch before any
median or favorable count. Report projected and lifted raw-space NRMSE/R2,
error versus horizon,
observed/predicted winding in identified invariant planes, eigenfrequencies,
performance versus `d`, and the gap to per-held-out-patch generators. A
360-angle nearest-orbit similarity search, if shown, is explicitly an in-sample
post-hoc alignment score and never a prediction metric. Orthogonality and
group-law residuals of `exp(theta A)` are numerical implementation checks only.
Controls are identity, a single shuffled-angle fit with seed `284733341`, the
same complete pipeline for the normal VAE, patch-specific generators, the
Procrustes cross-check, and the spatial action prescribed for SO(2)/used as a
candidate control for normal.

The per-patch overfit control uses the same fixed shared basis, its own
held-out-patch 5-degree anchors, the same 0-degree rollout and the same withheld
scoring, and is never called transferred evidence. For
each `d`, repeat the shared fit while leaving out each of the 17 fit patches and
report eigenfrequency matching/stability plus fit-patch rollout distributions.
Select one model-independent descriptive `d*` without held-out data as the
smallest `d` within one fit-patch standard error of the lowest mean of the two
models' leave-one-fit-patch-out median lifted NRMSE values; the complete
`d=1,...,6` held-out curve is always reported and both models use the same `d*`.

`d=1` is the invariant/null baseline; `d=2` is the first nontrivial rotation
plane. Interpret only eigenfrequencies, invariant-plane dimensions, transfer,
variance coverage and prediction errors, never individual matrix entries.

Primary fixed-sample decision rules are preregistered. At `d*`, evidence for a
transferred action in one model requires: median held-out strict one-view lifted
NRMSE at least 10% lower than both identity and shuffled controls; median lifted
R2 above zero; improvement over identity in at least 6/8 held-out patches; and
no leave-one-fit-patch-out frequency mode changing by more than 20% after
optimal frequency matching. An `SO(2)`-specific advantage additionally requires
its median paired NRMSE to be at least 10% lower than the normal VAE and to favor
`SO(2)` in at least 6/8 patches. Values within the 10% margin are treated as
equivalent for interpretation. Failure of these gates means a shared action is
unresolved/rejected for this fixed sample, even if coarse-orbit-conditioned or
local re-anchored prediction looks good. These are descriptive fixed-sample
margins, not significance tests. All other dimensions, metrics and controls are
secondary exploratory results.

## Experiment 7 Prescribed/Candidate Action And Factorization

For each model separately, measure corrected dense spatial-candidate-action
residuals and canonicalize full spatial posterior fields by its inverse. This
action is architecturally prescribed for the SO(2) VAE; for the normal VAE it
is only the same prespecified spatial control. Report the exact-quarter residual
medians unchanged beside the corrected dense result. For canonicalized vectors
`q_i(theta)` of scalar dimension `D`, define
`bar q_i=mean_theta q_i(theta)`, `bar q=mean_i bar q_i`,
`W=mean_(i,theta)||q_i(theta)-bar q_i||^2/D`, and
`B=mean_i||bar q_i-bar q||^2/D`. Report:

- within-orbit variance `W`, between-patch variance `B`, and `F=W/(B+eps)`;
- identity-control `W`, `B`, and `F`, plus `W/W_identity` and `B/B_identity` so
  an inflated denominator or collapsed code cannot masquerade as separation;
- held-out-angle nearest-centroid patch retrieval, with prototypes built only
  from each held-out patch's 5-degree anchors and accuracy first computed per
  patch;
- pairwise-distance stress
  `sqrt(sum_(a<b)(d_ab(theta)-d_ab(0))^2 / sum_(a<b)d_ab(0)^2)`, averaged over
  withheld angles, with theta=0 as the fixed reference; if its denominator is
  `<=1e-8`, stress is undefined and the content-preservation gate fails;
- a linear circular angle probe fitted on fit-patch 5-degree anchors to predict
  `(cos theta,sin theta)`, evaluated on held-out patches/withheld angles using
  mean circular error and mean cosine alignment
  `mean cos(wrap(theta_hat-theta))`; a second probe with the identical algorithm
  is fitted from scratch on canonicalized fit-patch anchors and evaluated on
  canonicalized held-out angles to measure residual leakage.

Learned-action canonicalization uses the fixed shared basis and the patch's
fit-angle orbit center; the untouched orthogonal residual is retained. It is
therefore a local, multi-view orbit analysis, not a deployable single-view
disentangler. Known-action and learned-action conclusions stay separate.

Learned-action H6 is evaluated only on the eight held-out patches; fit-patch
values are labeled in-sample diagnostics. The prescribed/candidate spatial
action requires no learned fit and retains a separate all-25 description.
Factorization support for the learned action additionally requires held-out
H4 to pass, identity-control `W_identity>=1e-6` and `B_identity>=1e-6`, and
pre-canonicalization angle-probe mean cosine alignment `>=0.50`. If any
nondegeneracy prerequisite fails, H6 is unresolved rather than supported.
Subject to those prerequisites, support requires held-out median patch-level
`W/W_identity <= 0.25`, `B/B_identity` within `[0.8,1.2]`, at least 90% mean
held-out retrieval, median distance stress no worse than identity by more than
10%, and post-canonicalization angle-probe mean cosine alignment at least 50%
lower than before. Meeting some but not all gates is "unresolved"; failure of
the within-variation or content-preservation gate is "not supported." These
thresholds are descriptive and do not imply population generalization.

## Experiment 8 Local Bundle And Gauge Diagnostics

Only when the fitted reduced representation contains one isolated rank-2
`m=1` plane may it define a scalar local phase: its frequency must lie in
`[0.8,1.2]`, be separated from every other positive eigenfrequency by at least
`0.25`, and its multivariate `m=1` energy must be at least 50% of the plane's
energy. Fix handedness by increasing angle, learn the plane/template on fit
patches, estimate each held-out patch's constant phase and amplitude only from
its 5-degree anchors, and score residual/phase on withheld angles. A repeated
`m=1` multiplicity is retained and described but cannot pass this gauge gate:
one orbit excites only one complex line, so a full orthogonal transformation in
the `U(k)` commutant is not identifiable from that patch's anchors. It would
require independent within-content perturbations not present in the fixed-25
protocol. If an isolated plane or its energy/separation conditions are absent,
report only descriptive orbit-shape homogeneity and make no gauge/bundle
compatibility decision.

Input stabilizer scores use disk-normalized self-distance
`s_i(theta)=RMS_D(x_i-R_theta x_i)/(RMS_D(x_i-mean_D x_i)+eps)`. A patch is
flagged only descriptively as near-isotropic if `max_theta s_i(theta)<0.10`, or
as possible `C2` if `s_i(180)<0.25` and is below half the mean of `s_i(90)` and
`s_i(270)`. Thresholds are fixed before outputs; all patches remain included.
Relate phase reliability to these continuous scores without claiming topology.
Define orientation strength `o_i=max_theta s_i(theta)` and C2 strength
`q_i=1-s_i(180)/(0.5[s_i(90)+s_i(270)]+eps)`. For the eight held-out gauge
scores, "failure concentrated in symmetric patches" requires either signed
Spearman `rho(phase_error,o_i)<=-0.5` or
`rho(phase_error,q_i)>=+0.5`; otherwise the association is unresolved. The
all-25 F1 phase diagnostic may report the same signed associations separately.
No angle- or copy-level inferential interval is allowed.

## Sampling Unit And Multiplicity

Angles and F1 copies are repeated measurements, not independent samples. All
population summaries and any resampling first reduce to one value per patch:
`n=25` for population/F1/spatial-action/stabilizer descriptions and `n=8` for
held-out shared-action, learned-factorization and gauge transfer. A paired
bootstrap, when used, resamples paired patches
and retains all angles/copies within each sampled patch. No p-values, angle-
level intervals, copy-level intervals, population claims, or causal claims are
permitted. Persist every one of the 25 or 8 patch values and favorable counts.

## Kaggle Authorization And Guard

The user explicitly authorized Kaggle for this correction and experiment on
2026-09-08. One private T4 inference-only launch is allowed after local tests,
kernel validation, independent spec/rotation review, and a fail-closed guard
bind the exact locked contract, payload, checkpoints, patch hashes, sources,
privacy, GPU, FP32, and internet-disabled settings. Guard phrase:
`spec0050_corrected_rotation_geometry_authorized`.

The launch may attach only the existing immutable checkpoint bundle and the
canonical pre-shuffled validation source. It may not train, write a dataset,
read sealed inputs, or reuse any Spec 0038 receipt/authority. One failed launch
may be retried only for a purely mechanical platform failure after an explicit
spec amendment and user authorization.

## Outputs And Acceptance Artifacts

Accepted output directory: `runs/local/corrected_rotation_geometry_v1`.

1. `figures/01-corrected-all25-orbits.png`.
2. `figures/02-population-input-controls.png`.
3. `figures/03-local-pca-ranks-00-12.png` and a companion PC1--PC6 sheet.
4. `figures/04-pc1-pc6-harmonics.png`.
5. `figures/05-all48-f1-summary.png`.
6. `figures/06-shared-generator-vs-dimension.png`.
7. `figures/07-canonicalization-factorization.png`.
8. `rotation_geometry_summary.json` with exact formulas, hashes, split, paired
   values, bootstrap draws/seed, F1 copy rows, generators/eigenfrequencies,
   factorization values, runtime, and output hashes.
9. Compact selected raw arrays sufficient to regenerate every figure: local
   PC1--PC6 scores for ranks 0/12, population per-patch metrics, F1 pooled
   traces/residual summaries, and shared-basis scores. Multi-gigabyte raw
   activations are prohibited.

The old v1 package is never overwritten. The new manifest names it under a
`supersedes` field and records the exact defect reproduction.

## Report Update Contract

Only after the corrected package and independent scientific reviews pass:

- update `scripts/build_professor_final_report.py` as source of truth and rebuild
  the existing DOCX/PDF paths;
- replace/supersede Figures 7, 8a, 8b and withdraw the old 25/25 claim unless it
  independently survives;
- add `Geometría de la acción rotacional en el espacio latente`, including a
  separate `Interpretación geométrica` that distinguishes architectural
  guarantee, empirical verification, learned organization, transfer, local
  bundle compatibility, contradictions, and speculation;
- label every new result fixed-validation, post-hoc, exploratory;
- update executive summary, synthesis, limitations, conclusion, glossary,
  traceability, and requirements appendix consistently;
- preserve sealed reconstruction, WSI, tissue, bootstrap, and supervised
  sections and source numbers unchanged;
- do not update the paper, thesis, Overleaf, or GitHub issues.

Immediately before authoring the DOCX/PDF, run the document/PDF artifact
operation markers once, then render the DOCX and PDF to PNG, inspect every page
at full size, run image/accessibility audits, and require matching final PDF
bytes from the accepted DOCX render.

## Acceptance Criteria

1. Independent scalar and packed-F1 hand-indexed fixtures prove positive
   orientation, component order and cardinal continuity at the quantitative
   bounds above; inverse/composition interpolation errors pass fixture bounds
   and are recorded on the fixed 25.
2. The corrected dense operator has one interpolation path and never calls
   `rot90`; the exact-quarter control remains separate. The legacy writer
   refuses the preserved Spec 0038 output directory and every existing package.
3. Every result uses only the two accepted checkpoint hashes, the exact ordered
   fixed 25, FP32, and integer angles 0--359.
4. Input-path controls, cardinal exclusions, 1/2/5-degree robustness, paired
   values/counts, and descriptive paired-bootstrap intervals are complete.
5. Rank 0/12 local PCA, tangents, basis-invariant harmonics, all 48 F1 copies,
   shared-basis coverage/conditioning/stability, strict one-view shared-action
   transfer, known/learned canonicalization, and stabilizer/gauge diagnostics
   meet the contracts above without cherry-picking or dropped patches.
6. Machine-readable results reproduce every plotted/report value and hash every
   output.
7. Rotation-correctness, mathematical-claim, statistical, and report-layout
   reviewers report no unresolved P0/P1; every accepted finding is verified and
   fixed.
8. Focused tests, touched-file Ruff/format/BasedPyright, full
   `./scripts/python_quality.sh`, `git diff --check`, repository preflight, and
   workspace preflight pass, or any unrelated pre-existing failure is reported
   exactly.
9. Final DOCX/PDF pass full-page visual inspection, accessibility/image audits,
   Letter/page-count checks, Spanish claim consistency, and output hashing.

## Tests And Verification Commands

```bash
.venv/bin/pytest -q tests/test_rotation_geometry.py \
  tests/test_frozen_vae_rotation_orbits.py tests/test_so2_vae.py
.venv/bin/ruff check src/eqvae/evaluation/rotation_geometry.py \
  src/eqvae/artifacts/rotation_orbits.py tests/test_rotation_geometry.py \
  tests/test_frozen_vae_rotation_orbits.py tests/test_so2_vae.py
.venv/bin/ruff format --check src/eqvae/evaluation/rotation_geometry.py \
  src/eqvae/artifacts/rotation_orbits.py tests/test_rotation_geometry.py \
  tests/test_frozen_vae_rotation_orbits.py tests/test_so2_vae.py
.venv/bin/basedpyright src/eqvae/evaluation/rotation_geometry.py \
  src/eqvae/artifacts/rotation_orbits.py tests/test_rotation_geometry.py
./scripts/kaggle_kernel.sh build kaggle/kernels/corrected_rotation_geometry
./scripts/kaggle_kernel.sh validate kaggle/kernels/corrected_rotation_geometry
./scripts/kaggle_kernel.sh check kaggle/kernels/corrected_rotation_geometry
./scripts/python_quality.sh
./scripts/agent_preflight.sh
../agent_preflight.sh
git diff --check
```

## Independent Preimplementation Review

Two clean-context adversarial reviewers independently examined rotation
correctness/provenance and the mathematical/statistical design. After revision,
both reported no unresolved P0/P1 and no lock-blocking P2. The final review
specifically accepted the corrected affine sign, packed-F1 fixture, two-phase
stale-consumer gate, strict one-view shared-action test, raw-space coverage,
held-out factorization, basis-invariant harmonics, and patch-level sampling.

## Accepted Run And Results

The single authorized private run completed as
`maximshtefan/eqvae-corrected-rotation-geometry/1`. Its receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-corrected-rotation-geometry/v0001.json`
and its downloaded package is
`runs/local/corrected_rotation_geometry_v1/corrected_rotation_geometry_v1`.
The accepted summary SHA-256 is
`aab68ca5b7cbe989ea1cd79740a64d83784441f85ffde09a6ee6ebebe7faf43c`;
`selected_arrays.npz` is
`c693d3ad26beab9da6c439654239851e760b41f730469e0d3ff39c7b37723ed1`.
All ten internal output hashes match the manifest and download receipt.

- Rotation fixtures passed on CPU and T4 at 32 and 256 pixels: cardinal error
  was zero, maximum sided-cardinal RMS was `5.20e-6`, maximum orientation error
  `0.0193` degrees, and inverse/composition error remained below `0.024`.
  Corrected rank-12 input steps around 90 degrees were
  `0.1011/0.1025/0.1118/0.1129/0.1035`; the old approximately `0.24` seams
  disappeared.
- H1 failed. At one degree, input-normalized local-linearity medians were
  `0.648894` normal / `0.601011` SO(2), favorable `25/25`, but the relative
  reduction was only `7.38%`. At five degrees the medians were
  `1.031667/1.032678`, favorable only `8/25` and slightly reversed.
- H2 failed. Low-harmonic fractions were `0.107456/0.106178`, favorable to
  SO(2) in `9/25`; effective-frequency medians were `4.134981/3.982210`, only
  `3.7%` lower despite `18/25`. Median PC1--PC6 capture was about `11%`.
- H3 failed: `0/48` internal F1 copies were clean; patch-median full-field
  residual was `1.9058` and median phase RMSE `1.6705` radians.
- H4 failed. Fit-only selection chose the null baseline `d=1`; held-out
  NRMSE/R2 was `1.420167/-1.016875` normal and
  `1.450213/-1.103134` SO(2), with `0/8` improvements over identity for both.
  Therefore H5's exact exponential group law is only imposed numerical
  consistency, not empirical action evidence.
- The prescribed/candidate spatial action was poor for both checkpoints:
  residual medians `1.675425/1.667622`, with `0/25 <= 0.25`.
- H6 remains unresolved rather than rejected: the learned action is identity
  at `d=1`, angle-probe prerequisites fail and canonicalization cannot test a
  nondegenerate factorization. H7 is unsupported: no eligible transferable
  `m=1` plane, stabilizer association or reliable local phase was found.

The exact-quarter archived medians remain the headline control unchanged:
normal `1.327492/1.303358/1.326518` and SO(2)
`1.472552/1.417673/1.486783`. The T4 normal remeasurement drifted by at most
`9.29e-5`; it is runtime provenance, not a replacement for the accepted values.

## Independent Post-Result Review

Clean-context mathematical and statistical reviewers found no P0 artifact
defect and accepted the negative decisions. Their only P1 was that the remote
summary pooled the H6 angle-probe rows over angles instead of persisting eight
patch units. Without new inference, the fixed `selected_arrays.npz` deterministically
reproduced all eight held-out rows in
`docs/data/spec0050_rotation_geometry_review_addendum.json` (SHA-256
`ec74b37f509205bca12294321844a3cfb48dbd2f1763c8ff94290fe064e86f94`);
the pooled value was unchanged and the unresolved conclusion remained. The
review also requires the selected-dimension leave-one-patch-out standard error
to be described only as a preregistered heuristic tolerance and the empty
`d=1` frequency result as not applicable. A clean-context layout reviewer then
found no P0/P1/P2 across all 26 DOCX/PDF pages and confirmed pixel-identical
separate rasterizations.

## Verification State

- Focused rotation/geometry tests: `59 passed` with two upstream deprecation
  warnings.
- Touched scientific Python files pass Ruff format/check and BasedPyright; the
  report builder compiles, executes and passes Ruff under the separate bundled
  document runtime.
- DOCX accessibility audit: high/medium/low `0/0/0`; 20 inline figures and all
  heading/section audits pass.
- DOCX and final PDF: all 26 Letter pages rendered and inspected without
  clipping, overflow, detached captions or illegible content.
- The repo-wide Python gate retains 23 unrelated pre-existing packaged-probe
  lint findings; full BasedPyright retains 31 unrelated pre-existing errors in
  the Spec 0037/0039 tests. Exact scope is recorded in `CURRENT.md`.

## Implementation Blockers

None. The Kaggle authority is consumed; no retry or further remote mutation is
authorized.

## Known Risks And Adversarial Checks

- Interpolation can manufacture smoothness. Compare latent metrics to the same
  input path, disk/full variants, cardinal exclusions, and subsampling.
- Orbit centering and patch-specific anchors are multi-view information. Keep
  learned canonicalization explicitly local/post-hoc; only the strict global
  rollout from the 0-degree embedding is primary shared-action evidence.
- PCA can mix harmonics and rotate signs. Prefer subspace power, frequencies,
  held-out prediction, and basis invariants.
- `exp(theta A)` obeys the group law by construction; do not present its near-
  zero residual as discovered evidence.
- Disk-pooled F1 vectors may cancel. Require full-field residuals.
- Small-amplitude/symmetric patches may not admit phase. Retain and explain
  them rather than filtering.
- Twenty-five fixed patches and one checkpoint per model do not support
  population or causal claims.
- Reviewers must try opposite sign, swapped F1 component order, degrees/radians
  confusion, angle leakage, train/held-out contamination, non-isotropic PCA
  axes, bootstrap resampling of unpaired values, and stale report prose.

## Open Questions

None within the fixed Spec 0050 scope. Higher-dimensional/nonlinear actions,
population generalization and training-seed robustness require a separate new
spec and authorization; they are limitations, not blockers.

## Related Files

- `docs/specs/0010-fixed25-equivariance-artifact-protocol.md`
- `docs/specs/0012-continuous-so2-vae-architecture.md`
- `docs/specs/0014-fixed-f01-full-vae.md`
- `docs/specs/0038-frozen-vae-rotation-orbit-visualization.md`
- `docs/specs/0040-spatial-latent-pca-coherence-visualization.md`
- `docs/specs/0044-professor-metrics-and-plots.md`
- `docs/specs/0048-professor-final-experiment-report.md`
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`
- `docs/data/spec0050_rotation_geometry_contract.json`
- `src/eqvae/artifacts/rotation_orbits.py`
- `scripts/build_professor_final_report.py`
