# Spec 0053: Functional And Riemannian Latent Geometry

Status: draft active; independently reviewed, lock blockers remain
Implementation readiness: not ready; bounded stage amendments and machine contracts pending
Owner/workstream: fixed-validation post-hoc functional, group-theoretic, and Riemannian audit
Last updated: 2026-09-10

## Purpose

Determine whether the frozen continuous-`SO(2)` VAE represents rotation through
a scientifically meaningful geometry that was missed by Euclidean PCA of the
flattened posterior mean. Compare it with the frozen normal VAE under the same
protocol. The workstream treats images and embeddings as spatial fields, uses
exact lattice-aligned `C4` as the interpolation-free anchor, audits a padded
continuous extension, studies the decoder-induced and posterior-distribution
geometries, and tests whether rotations and content changes follow transferable
low-complexity actions or coherent geodesic paths.

This is a new post-hoc, validation-only program. It may falsify every proposed
structure. It does not seek a positive `SO(2)` narrative.

The motivating accepted facts are:

- the final posterior tensors are spatial scalar fields
  `mu, logvar in R^(16x32x32)`, not 16 unstructured scalar coordinates;
- Spec 0050 found no transferable action in a generic Euclidean PCA basis of
  dimension at most six and no local factorization;
- Spec 0051 found an almost exact, parameter-free decoder action for exact
  `rot90/180/270` in the `SO(2)` checkpoint: aggregate action ratio
  `0.00000218`, versus `0.709453` for the normal VAE;
- the same `SO(2)` decoder was only descriptively better on the interpolated
  dense action (`0.549565` versus `0.712610`) and missed the locked `0.50`
  absolute rule;
- raw Euclidean `mu` residuals and end-to-end input commutation do not provide
  a corresponding general `SO(2)` advantage.

The central possibility is therefore not necessarily a fixed low-dimensional
Euclidean pose subspace. It may instead be a spatially distributed action with
large linear span, low group-harmonic complexity, and a simpler geometry after
accounting for the decoder and posterior uncertainty.

## Hard Boundaries And Non-Goals

- Do not retrain, fine-tune, mutate, convert, or select either VAE checkpoint.
- Do not access sealed test images, latents, labels, predictions, or scores.
- Do not alter the accepted sealed reconstruction, WSI diagnosis, tissue,
  bootstrap, or supervised-development conclusions.
- Use only the exact SHA-pinned fixed validation 25 and the accepted frozen
  60,000-update checkpoints.
- Do not tune a method from the normal-versus-`SO(2)` comparison outcome. A
  numerical pilot may choose a viable implementation only through
  model-blind reconstruction, residual, conditioning, and runtime gates.
- Do not call an orbit smooth merely because it is closed, or call a geodesic
  semantic merely because it minimizes a chosen energy.
- Do not infer a global manifold, global section, global product, bundle
  topology, disentanglement, causal architectural effect, population effect,
  or training-seed robustness from 25 validation patches.
- Do not identify `C4` frequency classes with unique continuous `SO(2)`
  frequencies. Exact `C4` observes angular frequency only modulo four.
- Do not call a covariance steerable after explicitly symmetrizing it and then
  use that constructed symmetry as evidence that the VAE learned it.
- Do not use UMAP-induced Riemannian PCA as a primary or required method. The
  mechanistic decoder pullback and analytic posterior metrics answer the
  relevant questions more directly. RPCA may be restored only by amendment
  with a distinct scientific purpose and validation contract.
- Do not construct or persist a full decoder Jacobian or full
  `J_D^T J_D` matrix at any native or expanded canvas size.
- Do not update the professor report, paper, thesis, Overleaf, GitHub issue,
  commit, push, or publish data under this spec. Accepted results may later be
  integrated through a separate result-bound reporting spec.
- Preserve Specs 0038, 0050, 0051, and 0052 and all accepted/superseded
  artifacts byte-for-byte.

## Authorization Boundary

The user authorized this scientific workstream and use of private Kaggle on
2026-09-10. This master roadmap is deliberately not a blanket launch authority:
the number and identity of remote jobs cannot be known safely until the
result-blind runtime and continuation probe has passed. It is not authority to
reuse a consumed guard, write any dataset, access sealed data, or launch before
a bounded stage amendment and its machine contract are locked.

The first executable unit must be a separate result-blind preflight amendment
authorizing exactly one unique-slug Kaggle launch and at most one continuation.
It may run analytic fixtures, size/runtime measurements, and predecessor-mount
tests, but it may not emit model-comparison results. Every later stage amendment
must declare its exact launch count, continuation ceiling, work-unit inventory,
owner-qualified kernel slugs, and guard phrases. The default ceiling is one
launch plus one continuation per amendment; exceeding it requires a new
amendment and fresh exact authorization. The conceptual A--F schedule below is
a dependency roadmap, not six preauthorized jobs.

Each remote stage or continuation must have:

- its own unconsumed fail-closed guard and exact stage contract;
- explicit `KAGGLE_PUSH_CONFIRMED=1` at the authorized push;
- an immutable owner-qualified launch receipt;
- a predecessor manifest when it consumes earlier stage output;
- private visibility, internet disabled, no model-parameter optimizer, and no
  training path; explicitly enumerated latent/chart/path optimizers may operate
  only with all frozen model parameters asserted byte-unchanged;
- a bounded deterministic work-unit set and no result-dependent automatic
  retry.

Remote reads and downloads remain receipt-bound and require
`KAGGLE_REMOTE_CONFIRMED=1`. A continuation processes only work units fixed in
the locked contract that did not complete before the preceding wall-time stop;
it is not a scientific retry or hyperparameter search.

## Fixed Inputs And Data Contract

| Item | Locked value |
| --- | --- |
| Normal checkpoint | `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`; SHA-256 `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075` |
| `SO(2)` checkpoint | `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`; SHA-256 `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7` |
| Fixed patches | `docs/data/fixed25/originals.pt`; exactly the existing 25 ordered validation patches and their accepted individual hashes |
| Input field | FP32 `3x256x256`, reconstructed as `uint8 / 255 * 2 - 1` |
| Posterior field | deterministic FP32 `mu, logvar`, each `16x32x32` |
| Sampled posterior | only where explicitly required; common reparameterization noise and recorded seed |
| Pre-head `SO(2)` field | `48F0 + 48F1 = 144x32x32`; F1 copy `j` uses packed channels `48+2j,49+2j` |
| Exact group | `C4={e,r,r^2,r^3}`, `r=torch.rot90(+1)`; exact `D4` remains a secondary extension |
| Continuous pilot | `0,10,...,350` degrees through one uniform positive-orientation operator |
| Dense continuation | `0,1,...,359` only after the padding/interpolation gate; 360 is a display-only copy of 0 |
| Native latent support | full `32x32` and centered radius-14 disk |
| Native input support | full `256x256` and centered radius-112 disk |
| Expanded primary canvas | input `384x384`, latent `48x48`, centrally aligned; `512/64` is a prespecified sensitivity only |
| Precision | FP32 model inference, JVP/VJP, and differentiable optimization; CPU FP64 reduced QR/SVD/eigendecomposition and final scalar accumulation |
| Biological comparison unit | WSI, `n=16`; patch rows, angles, group elements, fields, channels, modes, graph edges, and Monte Carlo samples are repeated or nested measurements |

No new input pair or displayed example may be selected from the new results.
Ranks 0 and 12 remain the primary qualitative patches. Cross-content displays
use the fixed pairs `(0,12)`, `(1,13)`, `(2,14)`, and `(3,15)`; every
population statement still uses all 25 patches.

The Spec 0050 patch split cannot be reused for transfer: five of its eight
held-out patches share a source WSI with fit patches. Spec 0053 instead locks a
WSI-disjoint split using only the pre-existing selector metadata. Sort the 16
unique WSI identifiers by

`SHA256("spec0053-wsi-split-v1:" || canonical_decimal_wsi_id)`

and assign the first five WSIs to evaluation. This result-independent rule gives:

- evaluation WSIs `5970,60988,38349,11417,10077`, ranks
  `3,2,0,1,11,6,7,9` (five WSIs, eight patches);
- fit WSIs `39880,46444,23629,22654,59031,27747,26025,31297,21260,39252,57265`,
  ranks `4,5,8,10,12,14,13,16,18,19,20,21,22,15,17,23,24`
  (11 WSIs, 17 patches).

The mapping and digest order come from
`configs/spec0001/fixed_25_validation_patches.json` and must be repeated in the
machine contract. “Fit” means fitting post-hoc bases or operators, never VAE
training. The eight evaluation patches are used only after every scientific
configuration is frozen. Nevertheless, all 25 patches were viewed in Specs
0050/0051, so this is a within-workstream WSI-disjoint transfer test, not an
untouched holdout, independent confirmation, or preregistration. The old split
may be reproduced only as a continuity sensitivity under that exact label.

## Mathematical Objects And Terminology

### Fields And Their Discretizations

In the continuum idealization, an RGB patch and a posterior mean are fields

`x : Omega_X subset R^2 -> R^3`,

`z=mu(x) : Omega_Z subset R^2 -> R^16`.

Candidate ambient Hilbert spaces are

`X=L2(Omega_X,R^3)` and `Z=L2(Omega_Z,R^16)`,

or Sobolev spaces such as `H1` when spatial derivatives are part of the metric.
The implemented networks operate on finite samples

`X_h ~= R^(3*256*256)` and `Z_h ~= R^(16*32*32)`.

There is no conflict between these descriptions. One field has a two-
dimensional domain; the vector representation has one scalar coordinate for
every sampled location and channel.

The ambient spaces are linear. The represented sets

`M_X subset X`, `M_mu=E_mu(M_X) subset Z`,

may be finite-dimensional manifold-like or stratified subsets, but smoothness,
dimension, and global manifold status are empirical questions. A tangent
perturbation at a latent field is itself a field

`xi : Omega_Z -> R^16`.

### Two Different Bundle Notions

The architectural feature-field bundle and the hypothesized pose bundle are
different objects.

1. A final F0 posterior is a section of the trivial rank-16 feature bundle over
   spatial coordinates. `SO(2)` acts on the base coordinates and trivially on
   the F0 fiber.
2. A content-pose bundle would be a local quotient structure
   `M_mu -> M_mu/SO(2)` whose fibers are represented rotation orbits. This is
   not guaranteed by the architecture and remains unproven.

Internal F1 fields have a nontrivial two-dimensional fiber representation, so
their action rotates spatial coordinates and vector components. This does not
by itself prove that the final posterior or data manifold forms a pose bundle.

### Domains And Exact Groups

A square grid is exactly preserved by `C4` and `D4`, not by arbitrary
continuous rotations. A centered disk is preserved by `SO(2)` in the continuum.
Zero-extending a field to `R^2` also defines an `SO(2)` action, but observing it
through a finite square window introduces support and sampling effects.

Consequently:

- exact `C4` is the primary interpolation-free group experiment;
- disk-restricted continuous analysis tests an invariant subdomain but omits
  exterior decoder information;
- expanded-canvas analysis approximates the zero-extended plane but is an
  out-of-training-size network evaluation that must pass an identity gate;
- native-square continuous interpolation is retained only as a control.

### Known Actions And Two Orbits

For a final F0 field, the prescribed spatial action is

`[rho_Z(theta)z](u)=z(R_-theta u)`.

For an internal F1 field,

`[rho_1(theta)h](u)=R_theta h(R_-theta u)`,

subject to the already verified repository sign convention.

For one patch define:

- encoded/data orbit
  `O_enc(x)={E_mu(R_theta x)}`;
- ambient prescribed orbit
  `O_act(x)={rho_Z(theta)E_mu(x)}`.

Encoder equivariance would make corresponding points agree. Decoder action
consistency can hold even when these orbits are distant in raw Euclidean
coordinates.

### Lie Derivatives And Infinitesimal Intertwining

Let `J=[[0,-1],[1,0]]`. For a sufficiently regular spatial F0 field,

`A_Z z = d/dtheta rho_Z(theta)z |_(theta=0)`

and, up to the verified sign convention,

`[A_Z z](u)=-(J u) dot grad z(u)`.

This is the Lie derivative along the rotational vector field. Internal F1
fields add the infinitesimal internal-vector generator. Define the analogous
input generator `A_X`.

The differential encoder and decoder intertwining questions are

`dE_x(A_X x) ~= A_Z E(x)`,

`J_D(z) A_Z z ~= A_X D(z)`.

These compare tangents, not merely finite endpoints.

### Functional Inner Products

The baseline field inner product is

`<z,w>_L2 = integral_Omega sum_c z_c(u)w_c(u) du`.

On a uniform complete grid it differs from flattened Euclidean dot product only
by quadrature scale. Functional analysis becomes substantively different when
it adds invariant support, nonuniform quadrature, derivatives, smooth bases, or
a non-Euclidean metric. The primary derivative-aware control is

`<z,w>_H1(alpha) = <z,w>_L2 + alpha <grad z,grad w>_L2`.

`alpha` values are fixed from grid spacing and synthetic unit tests rather than
chosen by model comparison. Report `alpha=0` and the prespecified positive
value; additional values are sensitivity only.

### Decoder-Induced Geometry

For decoder `D`, define the finite-distance pseudometric

`d_D(z1,z2)=RMS(D(z1)-D(z2))`.

The local pullback form is

`g_D,z(xi,eta)=<J_D(z)xi,J_D(z)eta>_X`,

with metric matrix `G_D(z)=J_D(z)^T J_D(z)` under the output `L2` metric. A
derivative-aware output metric `W_X` gives

`G_D,W(z)=J_D(z)^T W_X J_D(z)`.

`G_D` is Riemannian only on a subspace where it is positive definite and
numerically well conditioned. Near-null decoder directions make it degenerate
or practically degenerate. The workstream must estimate this before using
Riemannian language.

For a basis `U in R^(16384xd)`, the restricted metric is computed without a
full Jacobian:

`G_U(z)=U^T J_D(z)^T J_D(z)U=(J_D(z)U)^T(J_D(z)U)`.

Riemannian calculations are defined only on an explicit reduced affine chart

`S_U(z_*)={z_*+Ua:a in R^d}`.

Endpoints are projected to this chart by a locked rule and every optimized knot
remains in it. If the thin operator `B(z)=J_D(z)U` loses its declared numerical
rank anywhere on a path, the chart does not support a Riemannian geodesic. An
unconstrained optimization over arbitrary latent tensors is instead called a
*decoder-energy path*; it is not evidence of a geodesic on `M_mu`. A claim about
geodesics on the represented set would require a separately validated local
parameterization of that set.

### Posterior-Distribution Geometries

The operational posterior is defined by the training/sampling implementation:
`ell=clamp(logvar_raw,-8,4)` and `sigma=exp(0.5 ell)`. Raw `logvar` is retained
only as telemetry; report clamp count/fraction per patch and fail a posterior
interpretation if a later machine contract's saturation gate is exceeded. For
diagonal Gaussian fields, the squared
Wasserstein-2 distance is

`d_W2(q1,q2)^2=||mu1-mu2||_L2^2+||sigma1-sigma2||_L2^2`.

The local Fisher metric per coordinate is

`ds_F^2=dmu^2/sigma^2 + 2 dsigma^2/sigma^2`.

Fisher arithmetic uses the clamped log scale and stable log-domain weights;
the machine contract fixes a sigma floor and reports every use of it rather
than allowing underflow to define an apparent low-rank direction.

These metrics answer whether apparent mean-field differences occur in
high-uncertainty coordinates and whether posterior uncertainty transforms with
the spatial action. They do not replace decoder geometry.

For a posterior tangent `delta q=(delta mu,delta sigma)` under the common-noise
reparameterization `z=mu+sigma odot epsilon`, the expected decoder metric is

`g_q(delta q,delta q)=E_epsilon ||J_D(mu+sigma odot epsilon)`

`(delta mu+epsilon odot delta sigma)||_X^2`.

using common random numbers across models and transformations. It is secondary
because it adds Monte Carlo error and cost.

Exact `C4` is special: its array operator is a permutation, so pushing forward
a diagonal Gaussian preserves diagonality. A non-cardinal interpolation matrix
`B_theta` instead produces covariance

`B_theta diag(sigma^2) B_theta^T`,

which is generally not diagonal. Therefore analytic diagonal W2/Fisher action,
geodesic, and PGA claims are primary only for exact `C4`. A continuous
re-diagonalized sensitivity may use

`mu'=B_theta mu`, `v'=(B_theta odot B_theta)v`,

but must quantify discarded off-diagonal covariance on tractable fixtures and
must be labeled an approximation, never the exact pushed-forward posterior.

### Quotients, Gauge, And Stabilizers

For exact `C4`, define the orbit class `[z]={rho(g)z:g in C4}`. If `rho` is an
isometry for the declared metric `d`, the quotient distance is

`d_Q([z_i],[z_j])=min_(g in C4) d(z_i,rho(g)z_j)`.

The minimizing group element gives a relative alignment `g_ij`. A local gauge
chooses one representative per orbit. Changing representatives by `h_i`
changes edge alignments as

`g_ij -> h_i g_ij h_j^-1`.

Cycle consistency tests whether

`g_ij g_jk g_ki ~= e`.

Representative independence is automatic for the Euclidean field metric and
exact-C4 diagonal W2 metric. Decoder and learned metrics must first pass the
locked action-isometry gate on the exact state/direction set. If they fail,
report either a one-sided alignment score or the symmetric orbit-set
dissimilarity

`delta_orb([z_i],[z_j])=min_(g,h) d(rho(g)z_i,rho(h)z_j)`;

do not call either a quotient metric or use it for quotient PGA.

For `C4`, the quotient identifies four points but removes no continuous tangent
direction. Discrete phase synchronization and a continuous mechanical
connection are separate constructions. A mechanical connection is defined
only on a free-action stratum with a validated continuous action, tangent
chart, group-invariant positive-definite metric, and
`g(A_Zz,A_Zz)` above a locked nondegeneracy floor. There, without an epsilon
that would break `omega_z(A_Zz)=1`,

`omega_z(xi)=g_z(xi,A_Zz)/g_z(A_Zz,A_Zz)`.

It is undefined below the floor. Such points may have weak orientation or a
nontrivial stabilizer and remain in the result as undefined, not regularized
phases. Ambiguous discrete edges retain all minimizers or use fixed predeclared
weights; they are never silently forced into one connection block.

## Evidence Hierarchy And Scientific Questions

The workstream preserves this hierarchy:

1. exact group-array identities and numerical closure;
2. finite endpoint decoder agreement;
3. group-harmonic concentration and low multiplicity;
4. differential intertwining through Lie derivatives;
5. a shared structured action that predicts WSI-disjoint evaluation patches
   and angles;
6. decoder-metric geodesics that preserve known content under pose change;
7. inverse-action or quotient stability that preserves between-content
   differences;
8. consistent local frames/transport compatible with a local bundle;
9. global product or topology, which remains out of scope.

The primary questions are:

- Q1: Is the exact `C4` behavior organized in a small number of shared
  representation sectors and radial/channel multiplicities?
- Q2: Does the `SO(2)` VAE have lower functional complexity than the normal VAE
  after respecting field structure, without relying on constructed symmetry?
- Q3: Do encoder and decoder differentials intertwine the known spatial Lie
  generator more faithfully for the `SO(2)` checkpoint?
- Q4: Does decoder geometry compress discrepancies that are large in raw
  `mu`, and is the pullback metric approximately invariant under exact `C4`?
- Q5: Do Wasserstein or Fisher posterior geometries expose a simpler action
  that mean-only Euclidean analysis misses?
- Q6: After padding and interpolation controls, does a continuous low-
  complexity action transfer to WSI-disjoint evaluation patches and angles?
- Q7: Does an unpenalized decoder-metric geodesic recover rotation, a fade, an
  off-manifold shortcut, or another transformation?
- Q8: Do constrained geodesics improve content/pose preservation over Euclidean
  latent lines without merely baking in the desired rotation?
- Q9: Does quotient alignment make cross-content interpolation and retrieval
  more stable?
- Q10: Are local alignment frames and transports cycle-consistent, and do their
  failures correspond to weak orientation or stabilizers?
- Q11: Does any reflection behavior form a coherent `D4` representation rather
  than isolated transform robustness?

## Falsifiable Hypotheses And Decision Language

All conclusions are prospectively locked *within a post-hoc fixed-validation
exploration*. This is not an independent preregistration because prior outputs
from all 25 patches are known. Non-fitted all-25 comparisons reduce patch rows
to 16 WSI medians; an `SO(2)`-specific advantage requires at least 10% median
improvement and the favorable direction in at least 12/16 WSIs. Fitted transfer
claims use only the five WSI-disjoint evaluation WSIs and require the favorable
direction in at least 4/5. Patch rows remain visible but do not inflate the
biological sample size. Values inside the margin are comparable.

Every method, priority, progression rule, capacity, optimizer budget, and
absolute gate must be serialized before Stage A. Scientific progression cannot
depend on model ordering. Only analytic fixtures, hardware limits, and a
model-label-masked worst-case numerical criterion pooled over both branches may
activate a predeclared fallback. Evaluation shards use anonymous model codes
until final adjudication. A method changed after any scientific output becomes
a new secondary exploratory version and cannot support the original hypothesis.

### H0: Discretization And Support Null

The exact `C4` advantage is a lattice-aligned decoder property, while apparent
continuous failures are substantially explained by support, interpolation, and
aliasing. Support requires the padded pilot to reduce the dense action error
relative to the native-square path without degrading the angle-zero identity
gate. Failure leaves the exact `C4` result valid and continuous extension
unsupported.

### H1: Group-Structured Functional Complexity

After exact `C4` decomposition and continuous harmonic validation, `SO(2)` uses
fewer effective group sectors/multiplicities or achieves better evaluation-WSI
reconstruction/prediction at matched functional rank. A low-dimensional claim
requires both retained raw-field energy and retained decoder-visible energy;
few modes that discard most observable variation do not pass.

### H2: Differential Equivariance

`SO(2)` has lower normalized encoder and decoder Lie-intertwining residuals than
normal under the same support and derivative discretization. Endpoint
commutation alone does not support H2.

### H3: Decoder-Metric Organization

The two candidate orbits may be far in Euclidean `mu` but close under
`d_D`/`g_D`, more strongly for `SO(2)`. Support also requires approximate metric
invariance under exact `C4`; arbitrary decoder contraction is not sufficient.

### H4: Posterior Statistical Organization

Including `sigma` under Wasserstein-2 or Fisher geometry reduces action
residual, harmonic complexity, or geodesic distortion for `SO(2)` relative to
normal without posterior collapse or unbounded uncertainty. Mean-only and
complete-posterior conclusions remain separate.

### H5: Shared Continuous Action

A structured action fitted on fit WSIs using exact `C4` anchors, or the
prespecified coarse-angle extension, predicts WSI-disjoint intermediate angles
better than identity, linear interpolation, shuffled-angle, and per-patch-fit
overfit controls. Exact `C4` identifies a continuous frequency only modulo
four. The
contract enumerates `m=q+4l`, plane pairings, and handedness within the fixed
band `|m|<=8` for every learned C4-only sensitivity. The primary analytic
Fourier--Bessel action instead has a prescribed spatial complex structure.
Independent non-cardinal transfer is required in either case.
Identity/inverse/composition
and integer-weight `2pi` periodicity of a matrix exponential/Fourier phase are
construction checks; predictive non-cardinal transfer is the scientific
evidence.

### H6: Rotation-Faithful Geodesics

For endpoints differing only by a known `C4` rotation, a decoder-metric path
preserves content and follows a monotone rotation more faithfully than a latent
line. The unpenalized result is reported first. A penalized path supports H6
only if its improvement transfers and is not a direct consequence of a penalty
that encodes the target orbit.

### H7: Local Content-Pose Separation

Quotient alignment or inverse action reduces within-patch pose variation while
preserving between-patch distances, patch retrieval, morphology, and posterior
uncertainty. A good-looking canonical image alone does not support H7.

For the exact-C4 primary, let `z_(i,g)=E_mu(R_g x_i)`,
`c_(i,g)=rho(g)^(-1)z_(i,g)`, `s_(i,g)=D(c_(i,g))`, and the single unrotated
prototype `p_i=D(E_mu(x_i))`. Every output is in normalized `[-1,1]` units and
is restricted to the centered radius-112 disk. Here
`||a-b||_M^2=mean_(u in M_X,c)(a(u,c)-b(u,c))^2` is masked pixel/channel MSE
and `d_M(a,b)=sqrt(||a-b||_M^2)` is masked RMS. Define

`s_bar_i=(1/4) sum_g s_(i,g)`,

`W_i=(1/4) sum_g ||s_(i,g)-s_bar_i||_M^2`,

`B_i=mean_(j:WSI_j != WSI_i) ||p_i-p_j||_M^2`,

`F_i=W_i/(B_i+1e-8)`.

Thus each patch has a within/between normalized row and patches are reduced by
WSI. A row is nondegenerate only when `B_i>=0.02^2=4e-4`; otherwise it remains
visible but cannot count favorable. For single-anchor retrieval, use only
`g=r,r^2,r^3` as queries and the masked RMS `d_M` to rank the 25 angle-zero
prototypes. For query `(i,g)`, let `T` be every prototype within `1e-8` RMS of
the minimum; its accuracy contribution is `1/|T|` if `i in T`, else zero.
Average the three queries within patch, take the median within WSI, and take the
mean of the 16 WSI values for the `>=0.80` gate. This tests recovery of a known
patch from one canonical view, not unseen-content classification. The global
distance-stress veto, using `d=d_M`, is

`sqrt(sum_(g,i<j,WSI_i!=WSI_j)(d(s_(i,g),s_(j,g))-d(p_i,p_j))^2 /`

`sum_(g,i<j,WSI_i!=WSI_j)d(p_i,p_j)^2)`.

It is defined only when the RMS baseline distance over the included prototype
pairs is at least `0.02`.

Mandatory controls are no inverse action and a fixed wrong-group inverse.
Latent functional, posterior, and learned-action versions are secondary and
retain their own units. A learned continuous version fits only on fit WSIs,
uses angle-zero evaluation-patch prototypes without evaluation-angle fitting,
and is never pooled with the known exact-C4 result.

### H8: Connection And Local Bundle Compatibility

Relative frames transfer with low cycle inconsistency, continuous transport is
stable on generic patches, and failures associate with weak orientation,
symmetry, or low tangent amplitude. This supports only local compatibility; it
cannot establish topology or a global product.

### H9: Emergent D4 Structure

Reflections form a coherent action only if all exact group relations,
composition paths, sector transformations, decoder commutation, and evaluation-WSI
transfer pass. One successful diagonal flip remains isolated robustness.

### H10: Cross-Content Curvature

For predetermined content pairs, latent lines, encoded input fades, and
decoder/posterior geodesics differ reproducibly. This is a geometry probe, not
a primary model-quality claim, because no unique semantically correct path
between different contents is known.

### Primary decision matrix

The following endpoints are the only primary decision-bearing quantities. All
other analyses and visualizations in this spec are ordered secondary
mechanistic or sensitivity analyses. `AR` means action residual divided by the
matched no-action/identity residual in the same model and the explicitly named
metric; decoded AR uses physical output units, whereas `W2-AR` uses posterior
distribution units. `NRMSE` denominators and every nondegeneracy floor are
fixed in the stage contract from analytic fixtures.

For decoded output, let `y_i=D(E_mu(x_i))`,
`y_i^act(theta)=D(rho(theta)E_mu(x_i))`, and
`y_i^ref(theta)=R_theta y_i`. All expanded-canvas outputs are centrally cropped
to `256x256`; `M_X` is the centered radius-112 disk and averages include its
three channels. Define

`r_act(i,Theta)=sqrt(mean_(theta in Theta,u in M_X,c)`

`(y_i^act(theta,u,c)-y_i^ref(theta,u,c))^2)`,

`r_id(i,Theta)=sqrt(mean_(theta in Theta,u in M_X,c)`

`(y_i(u,c)-y_i^ref(theta,u,c))^2)`,

`A_i(Theta)=r_act(i,Theta)/(r_id(i,Theta)+1e-8)`.

This is dimensionless; component RMS values use normalized `[-1,1]` output
units. H0 uses `Theta={10,20,...,350}` with the locked uniform interpolation
operator and H3 uses exact `Theta={90,180,270}`. H3 exactly inherits Spec
0051's aggregate-disk `A_i`. A patch is decoded-action-nondegenerate only when
`r_id>=0.02`; otherwise its row remains visible but cannot count favorable.

For H4, let `(mu_i(theta),sigma_i(theta))` be the clamped diagonal posterior of
`R_theta x_i`, `M_Z` the radius-14 latent disk, and use exact
`Theta={90,180,270}`. Define

`r_W2,act^2=mean_(theta,u in M_Z,c)[(mu_i(theta)-rho(theta)mu_i(0))^2`

`+(sigma_i(theta)-rho(theta)sigma_i(0))^2]`,

`r_W2,id^2=mean_(theta,u in M_Z,c)[(mu_i(theta)-mu_i(0))^2`

`+(sigma_i(theta)-sigma_i(0))^2]`,

`W2-AR_i=sqrt(r_W2,act^2)/(sqrt(r_W2,id^2)+1e-8)`.

The mean-only control removes the two sigma terms. The complete-posterior row
is nondegenerate only when `sqrt(r_W2,id^2)>=1e-4` latent RMS units. Angles and
scalars are aggregated before the square root; ratios are then reduced by
patch and WSI. No continuous re-diagonalized quantity enters these formulas.

| Hypothesis | Primary endpoint and domain | Absolute functionality gate | Comparative/support rule | Unit and contradiction veto |
| --- | --- | --- | --- | --- |
| H0 | Change in 10-degree decoded spatial-action `AR` from native square to the first eligible expanded canvas | angle-zero median discrepancy `<=0.05`, maximum `<=0.10`; inverse/composition floors pass | support for a discretization explanation if padded `AR` is at least 10% lower in both models; otherwise continuous support remains unchanged/unresolved | 16 WSI medians; any model/domain failing identity is ineligible, not unfavorable |
| H1 | Per-evaluation-WSI minimum rank in the nested exact-C4 functional-sector basis attaining raw-field energy `>=0.90` and decoded truncation NRMSE `<=0.10` | a solution must exist by rank 64 | the nested basis/order is fitted once on fit WSIs; SO2 evaluation rank is at least 10% lower in 4/5 evaluation WSIs without retuning | evaluation WSI; 11 fit-WSI rows are in-sample diagnostics; veto if retained decoder-visible energy is `<0.90` |
| H2 | Disk `L2` normalized encoder and decoder Lie residuals `(epsilon_E,epsilon_D)` | both SO2 medians `<0.25`; both generator norms at least 10 times fixture noise | both residuals at least 10% lower for SO2 in 12/16 WSIs | WSI; undefined weak-tangent cases remain and cannot count favorable; either failed arm means H2 not supported |
| H3 | Exact-C4 decoded action ratio plus restricted-metric isometry residual | SO2 action ratio `<=0.10` and isometry residual `<=0.10` on all declared direction classes | SO2 action ratio at least 10% lower in 12/16 WSIs | WSI; failed isometry forbids quotient/group-isometry interpretation, not positive-definite decoder-chart geodesics |
| H4 | Exact-C4 complete-posterior W2 `AR` relative to mean-only W2 `AR` | complete-posterior median `AR<=0.50`; clamp/sigma gates pass | at least 10% within-SO2 reduction and at least 10% SO2-vs-normal advantage in 12/16 WSIs | WSI; continuous re-diagonalized results and Fisher are secondary and cannot rescue failure |
| H5 | Primary `FB64` shared action field NRMSE on non-cardinal angles in the five evaluation WSIs | NRMSE `<=0.50`, `R2>=0.50`, decoded NRMSE `<=0.25` | at least 20% better than identity and 10% better than matched normal in 4/5 WSIs | evaluation WSI; `FB64` is fixed below; periodicity is a construction check; per-state decoder lifts excluded |
| H6 | Integrated physical-rotation output NRMSE for the unpenalized `d=32` path from projected `E(x)` to projected `rho(90)E(x)` versus the line with the same endpoints | chart positive definite on every knot; monotone angle on `>=0.90` of frames; content-drift ratio `<=0.25` | a target-free start must give at least 10% lower NRMSE in 4/5 evaluation WSIs; regularized/oracle-start paths cannot rescue it | evaluation WSI; chart/solver failure is unresolved; target-seeded-only recovery is unresolved |
| H7 | Per-patch exact-C4 decoded canonicalization ratio `F_i=W_i/(B_i+eps)` defined below | median `F<=0.25`, single-anchor retrieval `>=0.80`, global distance stress `<=0.20` | SO2 `F_i` at least 10% lower in 12/16 WSI reductions | WSI plus global veto; known/learned actions separate; continuous claim requires H5 |
| H8 | Fraction of covered unambiguous exact-C4 graph triangles with identity cycle product, under an isometric metric | at least 10 unambiguous triangles spanning at least 20/25 nodes; fraction `>=0.90`; gauge test passes | local compatibility only; no SO2-superiority claim from one graph | one fixed-set global diagnostic with leave-one-WSI-out refits; insufficient coverage is unresolved |
| H9 | Joint exact-D4 maximum decoded action ratio over all eight elements using the prespecified even-scalar output extension | median `<=0.10`, every-element median `<=0.20`, array group table exact | SO2 at least 10% lower than normal in 12/16 WSIs and all group paths pass | WSI; any failed relation yields transform-specific robustness only |
| H10 | Arc-length-matched path disagreement for the four fixed content pairs | all endpoint, refinement, and finite checks pass | descriptive only; no supported/not-supported model-quality claim | four prespecified pairs; no inferential count |

The exact-C4 decoder endpoint ratio in H3 and the already measured isolated
reflection behavior feeding H9 are inherited Spec 0051 prerequisites. Their
reproduction verifies protocol identity but is not a new discovery; new H3
evidence is the directional metric-isometry test, and new H9 evidence requires
the complete D4 gate.

A hypothesis is **supported** only when its absolute and comparative rules pass
and no veto fires. It is **not supported** after a numerically valid eligible
test fails a scientific threshold. It is **unresolved** only when eligibility,
identifiability, signal, metric, or solver validation fails. Secondary analyses
may explain a result but cannot reverse this decision.

## Stage 0: Numerical And Resumability Preflight

Before scientific output:

1. Revalidate checkpoint, state-dict, fixed-patch, operator, and model source
   hashes inherited from Specs 0050 and 0051.
2. Prove that all evaluation paths run under `torch.inference_mode` except
   explicit JVP/VJP or path optimization; assert zero optimizer creation for
   model parameters and zero parameter mutation.
3. Test every exact `C4` and `D4` array action on asymmetric scalar and vector
   fixtures, including group multiplication tables.
4. Test Lie-generator signs on analytic scalar/vector fields and compare the
   analytic derivative with central angle differences as the step tends to
   zero.
5. Test `L2`, `H1`, quadrature, disk, Fourier/group projectors, generalized
   PCA, randomized SVD, posterior metrics, path-energy, quotient, and graph
   routines on analytic problems with known answers.
6. Run eager FP32 first and validate first- and second-order autodiff, finite
   differences, adjoints, HVP/double-backward, and one Exp/Log/transport step.
   Compile only fixed-shape forward/JVP/HVP kernels that separately pass
   eager-versus-compiled closure. Lock backend/mode, warmup count, cold compile
   budget, cache policy, allowed graph breaks/recompiles, and eager fallback.
7. Pin deterministic algorithms, CuDNN deterministic/benchmark settings,
   matmul/TF32 policy, compiler settings, CUDA/GPU capability, and software
   hashes. A checkpoint may resume only on an exactly compatible environment;
   otherwise restart only the incomplete work unit. Floating artifacts require
   numeric tolerance plus exact contract/checkpoint/work hashes, not byte
   identity.
8. Benchmark both models under anonymous labels on one patch, one `C4` orbit,
   one padded continuous batch, and one 17-knot path iteration. Ramp JVP
   direction microbatches
   `1,2,4,8,...,32`, recording peak allocated and reserved memory, and keep at
   least 25% VRAM headroom; do not probe 32 first. Lock the common work size and
   call budget from the slower/larger masked branch and publish only masked
   telemetry before adjudication. Benchmark second-order and transport work
   separately because first-order timing cannot predict it.
9. Lock each bounded amendment's work-unit sizes, operator-call limits, memory,
   wall-time reserve, session-output size, predecessor-mount size, local
   download size, exact mandatory pilot, and result-blind expansion gate.
10. Prove continuation with a unique immutable kernel slug per session. The
    child embeds the expected predecessor owner/slug/version and manifest SHA,
    verifies the mounted manifest before work, and records the complete
    dependency DAG/all ancestor locators. Test same-stage continuation,
    multi-predecessor fan-in, older-shard access, and final local aggregation.
    If exact private predecessor output cannot be attached and authenticated,
    stop; do not base64/embed large state or publish a continuation dataset
    without separate dataset-write authorization.

## Stage 1: Padding, Support, And Sampling Audit

Define central padding/cropping operators `P_X,C_X,P_Z,C_Z`. The primary
expanded geometry maps `256->384` and `32->48`; `512/64` is sensitivity.

### 1A. Input-size identity

At angle zero measure

`C_Z E_384(P_X x)` versus `E_256(x)`

for `mu` and `logvar`, and

`C_X D_48(P_Z z)` versus `D_32(z)`.

Report raw and normalized `L2`, `H1`, SSIM, boundary-ring error, channelwise
error, output range, and GroupNorm/FixedF01FieldNorm activation-statistic
changes. Because normalizers aggregate over spatial extent, fully convolutional
structure does not imply padding invariance. Eligibility is assessed separately
per model/domain, but a paired claim requires the same route to pass for both.
A padded route is primary-eligible only if it is finite for all 25 patches and
its angle-zero median normalized discrepancy is at most 0.05 with no patch
above 0.10. Test `384/48` first; run `512/64` only if it passes. If either model
fails, native square/disk remains the common primary domain and padding is
labeled out-of-size/OOD rather than used for a continuous-action claim.

### 1B. Exact-group preservation

Exact `C4` algebra must close to its array-numerical floor within each canvas.
Across canvas sizes, compare transform-specific excess error after subtracting
the measured angle-zero size-change baseline; do not demand raw equality that
GroupNorm makes structurally implausible. This detects parity, coordinate
center, transposed-convolution, and crop alignment mistakes.

### 1C. Support variants

Compare:

- native square with zero extension;
- native centered disk;
- `48x48` zero-padded latent and `384x384` input;
- the `64/512` sensitivity canvas;
- a fixed apodized-boundary sensitivity.

Reflect/circular padding is not a primary physical assumption. It may be shown
only as sensitivity because tissue outside the observed patch is unknown.

### 1D. Interpolation and aliasing

Padding does not remove sampling alias. Compare the corrected bilinear operator
with one independently verified antialiased/band-limited candidate on analytic
fixtures and raw inputs. Do not select the operator from latent model results.
Record inverse, composition, energy, cardinal-neighborhood, and spectral
leakage errors. One operator is then locked for the continuous pilot; the
existing corrected bilinear path remains a required control.

## Stage 2: Exact C4 Field And Representation Analysis

For every model and patch compute exact fields at `0,90,180,270` for:

- input RGB;
- posterior `mu`;
- posterior `logvar` and `sigma`;
- decoded encoded-input states;
- decoded prescribed-action states;
- internal final-D F0/F1 fields where applicable.

For a four-state field orbit `z^(k)`, define the group Fourier components

`z_hat_q=(1/4) sum_(k=0)^3 exp(-2 pi i q k/4) z^(k)`,

`q=0,1,2,3`.

For the spatial action `rho(r)`, use the complex projector

`P_q z=(1/4) sum_(k=0)^3 exp(+2 pi i q k/4) rho(r)^k z`,

so `rho(r)P_q z=exp(-2 pi i q/4)P_q z`. With the response DFT sign above and
`rho_theta z(r,phi)=z(r,phi-theta)`, response class `q` corresponds to spatial
mode `m=-q mod 4`. The contract stores this map explicitly. An asymmetric
complex phase fixture must fail if `q=1` and `q=3` are swapped; pooling their
energy is not a sufficient sign test.

For real fields, `q=1` and `q=3` form a conjugate two-dimensional sector. The
analysis reports `q=0`, `q=2`, and joint `q=+/-1`; it never names these unique
continuous frequencies.

Perform two distinct decompositions:

1. **Response decomposition:** Fourier transform along the physical input-
   rotation index of `E(R_(90k)x)`.
2. **Spatial representation decomposition:** apply exact `C4` projection
   operators to one field under the canonical coordinate action.

Their agreement is an equivariance diagnostic. Report sector energy, phase,
cross-sector leakage, stable/effective rank, radial/channel multiplicity, and
WSI-disjoint evaluation reconstruction. Repeat on full square and disk.

Compute the raw covariance commutator

`||C rho(g)-rho(g) C||_F/(||C||_F+eps)`.

Also show a group-averaged covariance for constructing a steerable basis, but
never use its by-construction commutation as learned evidence.

Decode modal reconstructions using:

- invariant `q=0` only;
- `q=0` plus joint `q=+/-1`;
- all sectors.

Report reconstruction loss against the complete latent decode and against the
appropriate rotated output. A compact modal reconstruction that discards
visible morphology is not a positive low-complexity result.

## Stage 3: Functional And Steerable PCA/SVD

### 3A. Exact-square C4 basis

Use eigenspaces/projectors of the exact array permutation. This path requires
no polar interpolation and is the primary `C4` functional basis.

Within each sector, form matrices across fit patches, channels, and spatial
multiplicity coordinates. Fit SVD/PCA only on fit patches and apply the frozen
basis to WSI-disjoint evaluation patches. Report:

- singular spectrum and cumulative energy;
- stable rank `||X||_F^2/||X||_2^2`;
- entropy effective rank `exp(-sum p_j log p_j)`;
- evaluation-WSI field reconstruction versus retained rank;
- evaluation-WSI decoder-visible reconstruction versus retained rank;
- principal angles and projector stability under leave-one-fit-WSI-out refits;
- model comparison at matched retained energy and matched rank.

### 3B. Fourier-Bessel/steerable disk basis

On the invariant disk, expand each channel in angular modes and radial basis
functions:

`z_c(r,phi)=sum_(m,l) a_(c,m,l) J_m(kappa_(m,l)r)e^(i m phi)`.

Boundary conditions and quadrature are scientific choices. The H5 primary is
the Neumann disk basis with radial wavenumbers satisfying
`J_m'(kappa_(m,l)R)=0` and the `m=0,kappa=0` constant mode included separately;
it permits a nonzero boundary trace. The Dirichlet roots
`J_m(kappa_(m,l)R)=0` are a mandatory sensitivity. This ordering is fixed
before results and never selected by model advantage. Record basis roundtrip
error, conditioning, quadrature weights, maximum frequency, boundary energy,
and energy omitted by radial/angular truncation.

Perform SVD/PCA inside each angular `m` multiplicity space over radial index and
channel. For real fields, every `m>0` contributes a sine/cosine pair. Report the
effective real dimension

`d_eff=q_0+2 sum_(m>0) q_m`.

### 3C. Functional products

Run the same fixed bases under:

- uniform/full-grid `L2`;
- disk/quadrature `L2`;
- prespecified `H1`;
- decoder-restricted inner products from Stage 6;
- posterior Wasserstein and Fisher products from Stage 7 where defined.

For a fixed positive-definite matrix `W`, generalized PCA is implemented by
PCA of `W^(1/2)(z-z_bar)` and mapping components back with `W^(-1/2)`. For a
position-dependent metric, do not pretend that one global generalized PCA is
exact; use local tangent analyses or PGA.

### 3D. Spatial regularity and covariance operators

Measure per patch/channel:

- `L2`, gradient, Laplacian, total-variation sensitivity, and `H1` energy;
- spatial Fourier/Fourier-Bessel spectral decay;
- energy by radial ring and boundary trace;
- rotation-energy density `||A_Z z(u)||^2`;
- covariance kernels across positions/channels and separability ranks;
- response-mode/spatial-mode phase coherence;
- F1 radial/tangential, divergence/curl, and reflection-parity diagnostics.

All scalar summaries are first reduced within patch. Channels, positions, and
modes are not independent samples.

## Stage 4: Continuous Pilot And Dense Harmonic Validation

Run exact independent input rotations and prescribed latent rotations on the
locked padded/native operators at `10`-degree spacing. Do not interpolate
between stored embeddings to manufacture observations.

For orbit angle `theta` and spatial polar angle `phi`, test the field identity

`z_theta(r,phi) ~= z_0(r,phi-theta)`.

Equivalently, a spatial mode `m` should acquire group-angle phase

`a_(i,m,l)(theta) ~= exp(-i m theta) a_(i,m,l)(0)`.

Report separately for fit and WSI-disjoint evaluation patches:

- harmonic energy and leakage across `m`;
- phase slope, wrapped phase error, amplitude stability;
- radial/channel multiplicity rank;
- cross-patch basis stability;
- full-field and decoder-visible residual;
- input-path-normalized residual;
- group composition and inverse interpolation floors;
- native, disk, padded, and apodized-support variants.

The `10`-degree grid has 36 samples and can resolve observed orbit frequencies
through the Nyquist class 18. The primary scientific displays retain at least
`m=0,...,8`; higher energy is pooled and reported.

Proceed to the one-degree sweep only if:

- Stage 1 padding identity is primary-eligible;
- continuous inverse/composition fixtures pass;
- no cardinal discontinuity appears;
- pilot spectra show no aliasing within the reported mode range;
- all 25 patches complete for both models;
- output/memory projections fit the declared remote budget.

If this gate fails, retain exact `C4` plus the `10`-degree audit as the final
scope and state that continuous `SO(2)` recovery is unresolved.

## Stage 5: Lie-Derivative And Tangent Analysis

Compute the known spatial generators `A_Xx` and `A_Zz` using an independently
tested derivative discretization on the disk, full native field, and padded
field. Compare against central finite-angle derivatives at shrinking steps.

Use JVPs to calculate

`v_enc=dE_x(A_Xx)` and `v_dec=J_D(z)A_Zz`.

Primary normalized residuals are

`epsilon_E=||v_enc-A_ZE(x)||/(||v_enc||+||A_ZE(x)||+eps)`,

`epsilon_D=||v_dec-A_XD(z)||/(||v_dec||+||A_XD(z)||+eps)`.

Report `L2`, `H1`, decoder-visible, radial, channelwise, and spatial error maps;
tangent norm; tangent-angle continuity; acceleration; curvature; and the
singular spectrum of tangents along each orbit. Tiny-norm tangents are retained
and flagged as possible weak orientation/stabilizer cases rather than divided
into unstable ratios.

For internal F1 fields, include the internal generator and compare:

- spatial-only derivative;
- internal-only derivative;
- full spatial-plus-internal Lie derivative.

This distinguishes incorrect fiber transformation from spatial sampling error.

## Stage 6: Decoder Pullback Metric And Randomized Linear Algebra

### 6A. Precision contract

The decoder Jacobian has roughly `196608 x 16384 = 3.22e9` entries at native
size: about 12.9 GB in FP32 or 6.4 GB in FP16 before workspace, and much larger
on padded canvases. It is never materialized at any canvas size. FP16 storage
therefore does not solve the real problem and loses the small singular values
that determine the metric's null directions. Primary JVP/VJP, randomized
probes, metric energies, and path gradients use FP32. Reduced matrices are
transferred to CPU and analyzed with FP64 rank-revealing QR/thin SVD.

FP16/autocast is telemetry-only in this spec. Any decision-bearing AMP path
requires a separate amendment and paired FP32 recomputation of every affected
work unit and scalar. This directly answers the implementation question: use
operator products and randomized linear algebra, not a half-precision dense
`J_D`.

### 6B. Randomized spectral analysis

Treat `J_D` and `J_D^T J_D` as linear operators exposed only through JVP/VJP.
Use a deterministic randomized range finder with fixed Gaussian/Rademacher
seeds, oversampling, and power iterations, followed by deterministic reduced
SVD. Full-operator probes are limited initially to exact-C4 states for ranks
0/12 and angle-zero states for all 25; structured restricted metrics cover all
25. Any expansion is a separately bounded result-blind amendment. The final
contract must specify rank targets, initially
`k=8,16,32,64`, oversampling `p=16`, and power iterations `q=2`; the numerical
pilot may increase `p/q` only from masked certificate failure, never model
outcome. The common worst-case `p/q` and operator-call budget then apply to
both models, and every approximation must independently pass its certificate.

For every approximation record:

- seed and probe distribution;
- operator call counts;
- orthogonality residual;
- Ritz/singular residuals;
- direct independent-probe estimate of
  `||(I-QQ^T)J_D||_F^2` with locked probe distribution/count, 99% confidence
  bound, relative-residual threshold, spectral residual bound, and maximum
  operator calls;
- total/captured Frobenius energy from an independent Hutchinson estimate with
  the same explicit confidence accounting;
- principal-angle stability across at least three seeds;
- agreement with exact SVD on synthetic and small reduced problems.

Use exactly 128 independent residual probes for every decision-bearing
certificate; there is no optional stopping. A smaller 32-probe result may be
logged as telemetry only. Failure of the fixed 99% bound is unresolved, not
permission to accept the subspace. The same fixed probe rule applies to
residual and total-trace estimates. Captured-energy certification uses the
upper 99.5% residual bound divided by the lower 99.5% total-trace bound; the
Bonferroni allocation gives at least 99% simultaneous coverage for the ratio.
A flat tail that has stable leading
angles but fails captured-energy certification does not pass. Do not average
singular vectors directly: align subspaces, average projectors if a consensus
subspace is required, and eigendecompose the mean projector. Report the three
individual seeds and instability instead of selecting the best seed.

Every implementation must lock both randomized-linear-algebra controls: sketch
width `s=k+p` (the number of low-dimensional random projection columns used by
the range finder) and independent replication count `r=128` for each
decision-bearing scalar certificate. Each projection column is a
full-dimensional vector in the fixed `16x32x32` latent space; `s` and `r` do
not change that latent dimension. For every locked trace or Frobenius-energy
target, aggregate the `r` per-probe quadratic forms by their sample mean and
report its fixed 99% confidence bound. This estimates declared scalar
properties of the otherwise infeasible dense `J_D^T J_D` product with
controlled uncertainty; it neither materializes nor claims an entrywise
reconstruction of that matrix. A reduced matrix product is allowed only after
the separately certified randomized subspace has been constructed. Do not
average singular vectors or bases: align subspaces and average projectors only
when a predeclared consensus subspace is required.

### 6C. Restricted exact metrics

For every functional/harmonic basis `U_d`, calculate `B=J_D U_d` by batched
FP32 JVP. Transfer `B` to CPU and use FP64 rank-revealing QR/thin SVD; form
`G_U=B^T B` only as a derived convenience, never as the rank certificate,
because a Gram matrix squares the condition number. Validate JVP adjoints and
finite differences and define retained rank relative to an empirical FP32
singular-value noise floor. Compare:

- Euclidean variance spectrum;
- `L2/H1` functional spectrum;
- decoder-visible spectrum;
- local condition number and near-null dimension;
- overlap of orbit tangents with sensitive and insensitive subspaces;
- normal versus `SO(2)` at matched rank.

### 6D. Metric invariance

For exact `g in C4`, test

`g_D,rho(g)z(rho(g)xi,rho(g)eta) ~= g_D,z(xi,eta)`

on fixed analytic directions, orbit tangents, functional principal directions,
and randomized directions. This is stronger than comparing decoded pictures.
An action cannot be called a decoder isometry if it contracts all tested
directions nonspecifically.

## Stage 7: Posterior Wasserstein, Fisher, And Expected-Decoder Geometry

For `mu`, raw/clamped `logvar`, and `sigma` under exact `C4`:

1. compute Wasserstein-2 orbit distances, sector energies, generalized PCA,
   action/canonicalization residuals, and geodesics in `(mu,sigma)` coordinates;
2. compute local Fisher norms of rotation tangents and action residuals;
3. report uncertainty magnitude, anisotropy across channels/space, boundary
   concentration, and any collapse/saturation indicators;
4. compare mean-only and complete-posterior rank/phase conclusions;
5. use common fixed reparameterization noise to test sampled-latent action
   without turning noise draws into independent samples.

The Wasserstein geometry of diagonal Gaussians is analytically tractable in
`(mu,sigma)` for exact `C4`. Fisher geometry is treated locally in an explicit
validated product chart with the locked sigma floor. Continuous posterior
analysis is limited to mean fields unless the explicitly labeled
re-diagonalized covariance approximation is used; it cannot decide H4.

The expected-decoder metric uses exactly 16 fixed common-noise draws for both
models and every paired state. Report Monte Carlo standard errors and the 99%
confidence width. There is no outcome-dependent escalation or seed selection;
a precision failure at 16 is reported unresolved or addressed by a later
amendment that recomputes both branches with the same larger fixed draw set.

## Stage 8: Shared Structured Actions

Fit and compare the following action families without a free
`16384x16384` matrix:

1. prescribed pure spatial action;
2. exact `C4` group-sector action;
3. Fourier-Bessel phase action within retained `m` sectors;
4. spatial generator plus shared `16x16` channel generator;
5. spatial generator plus a small local convolutional correction;
6. one deterministic shared decoder-aware map fitted only on fit WSIs, if its
   family and selector are locked before evaluation;
7. per-state decoder preimages from Stage 9 as an oracle/nonuniqueness
   diagnostic that is excluded from H5;
8. generic Euclidean shared subspaces from Spec 0050 as the negative/control
   baseline.

The sole H5 primary family is `FB64`: the disk Neumann Fourier--Bessel
basis with analytically prescribed sine/cosine complex structure and
handedness, angular cutoff `|m|<=5`, within-sector fit-WSI SVD multiplicities
`q_0=4` and `q_m=6` for `m=1,...,5`, giving
`4+2*(5*6)=64` real coordinates. The basis is fitted only on exact-C4 states
from the 11 fit WSIs; every phase convention is fixed by the analytic spatial
basis, not by response phase. It acts by
`a_(m,l)->exp(-i m theta)a_(m,l)`, with no channel generator or learned
correction. All other families are secondary in the listed order and cannot
rescue H5.

The primary anchor-only protocol fits radial/channel templates on fit-WSI exact
`C4` states only. It predicts intermediate angles on evaluation WSIs without
seeing their non-cardinal views. The analytic Fourier--Bessel spatial basis
identifies its own integer `m`, complex structure, and handedness independently
of the response orbit. A learned real C4 multiplicity sector does not: `q=2`
leaves the tie `m=+/-2` and supplies neither a plane pairing nor handedness,
while `q=0` has higher-frequency aliases. Every nonprimary C4-only extension
must serialize `m=q+4l`, all real-plane pairings, complex structures, and
handedness within `|m|<=8` as explicit priors. Without that serialization, it
is non-identifiable until non-cardinal fit data are used and cannot support an
anchor-only claim.

A secondary coarse protocol may fit the `10`-degree fit-WSI states and must
evaluate the complementary angles and all evaluation WSIs. Report the
gap between anchor-only and coarse-continuous fits.

Both models receive the same family, dimension, parameter count, initialization,
optimizer evaluations, stopping rule, and convergence criterion. Metrics are
reported as within-model ratios against matched identity/known-action controls
and in common decoded-image units; raw model-specific latent scales are not
directly ordered. Every learned family is tested for:

- identity, inverse, periodicity, and composition as implementation checks when
  imposed by construction, and as empirical checks only when not imposed;
- orthogonality/isometry under each relevant metric;
- evaluation-WSI patch/angle prediction NRMSE and R2;
- winding and phase error;
- full-field and decoded prediction;
- retained Euclidean and decoder-visible energy;
- sensitivity to dimension `8,16,32,64` and harmonic cutoff;
- leave-one-fit-WSI-out subspace/frequency stability;
- gap to per-patch overfit, identity, shuffled angles, latent line, and the
  prescribed spatial action.

One action shared across WSIs is required. Patch-specific phase origins may
be estimated only in an explicitly local-gauge analysis and cannot support the
shared single-view action claim.

## Stage 9: Decoder-Preimage Analysis And Regularization Ladder

For target output rotation `R_theta D(z)`, first analyze the unregularized
decoder-preimage problem

`T_D,theta(z)=argmin_(z') ||D(z')-R_theta D(z)||_X^2`.

The mathematical object is the prescribed finite candidate solution set from
all converged starts, not automatically a single-valued map or group action:
the infimum may be unattained, minima may be nonunique, and local minima are
solver-dependent. Use fixed endpoints/angles, multiple predetermined
initializations
`z`, `rho(theta)z`, `E(R_theta x)`, the previous-angle solution, and seeded
small perturbations. Persist every solution and report nonuniqueness, latent
distance, posterior plausibility, and decoded agreement. Do not select the
prettiest minimum. Target-informed starts and per-state reoptimization are
oracle diagnostics and never count as evaluation prediction.

Then apply the following cumulative or factorial sensitivity ladder, with
weights scaled from fit-only dimensionless baseline energies:

1. latent/prior magnitude `R_prior=||z'||^2`;
2. local Mahalanobis distance using posterior `sigma`;
3. distance to the represented/orbit tube `R_M=d(z',M_mu)^2` estimated only
   from fit states;
4. temporal constant-speed penalty over a complete path;
5. Euclidean then covariant acceleration penalty;
6. closeness to the prescribed spatial action, explicitly a biased control;
7. content-coordinate preservation, only after content coordinates are fixed
   on fit patches.

Only a deterministic shared selector/map fitted on fit WSIs without
evaluation-target encodings may become a Stage-8 candidate. Its architecture,
tie-breaking, optimization, and failure behavior must be locked before
evaluation. When fitting that shared family `T_theta`, add separate group
penalties:

`T_0(z)=z`, `T_-theta(T_theta(z))=z`, `T_2pi(z)=z`,

`T_alpha(T_beta(z))=T_(alpha+beta)(z)`.

Composition is not meaningful as a penalty on one independently optimized
endpoint. Report reconstruction objective and every penalty separately so a
large regularizer cannot masquerade as decoder agreement.

## Stage 10: Geodesics For Known Transformations

Primary Riemannian paths live in the shared fit-WSI functional chart
`S_U(z_*)` with `d=32`; `d=8,16,64` are fixed sensitivities. For each endpoint
pair, `z_*` is their Euclidean midpoint and the endpoints are orthogonally
projected into the frozen shared basis. Report projection error before any path
comparison. An otherwise identical unrestricted-latent solve is labeled a
decoder-energy shortcut diagnostic. Represent a path by fixed knots
`gamma_0,...,gamma_K` with fixed endpoints.
The primary discrete decoder energy is

`E_D(gamma)=sum_(k=0)^(K-1) ||D(gamma_(k+1))-D(gamma_k)||_X^2/delta_t`.

Its infinitesimal limit is the pullback energy where the chart metric is
positive definite. Optimize first with no off-manifold or smoothness penalty.
The mandatory pilot uses `K=16`. Primary starts are the latent line plus two
seeded target-free perturbations of that line. Prescribed-action and encoded-
physical-rotation starts are oracle/basin sensitivities excluded from H6. The
exact matched optimizer and operator-call cap are locked by the result-blind
amendment. A `K=32` prolongation is mandatory for every accepted primary path;
`K=64` is ranks-0/12 sensitivity only. A primary recovery must arise from a
target-free start; if only a target-informed start reaches it, or target-free
and target-informed starts disagree beyond the convergence tolerance, H6 is
unresolved. Failure at the cap is completed, not permission to change solver.

Run ranks 0/12 as the mandatory numerical/visual pilot. If the model-blind
conditioning, convergence, reversal, and refinement gate passes, the primary
H6 stage runs all eight patches from the five WSI-disjoint evaluation WSIs.
All-25 decoder-geodesic and decoder-PGA expansion is a separate bounded
amendment, never an “if affordable” post-result choice. H6 uses the chart
projections of `E(x)` and `rho(90)E(x)` as its single primary endpoint pair;
the geodesic and Euclidean-line control share those endpoints and compare to
the physical output reference `R_90 D(E(x))`. The other pairs are secondary:

- `E(x)` and `E(R_90 x)`;
- unprojected `E(x)` and `rho(90)E(x)` as an ambient decoder-energy control;
- adjacent exact `C4` anchors;
- `E(x)` and the relevant exact reflection as secondary `D4` analysis.

Candidate paths are:

- Euclidean latent line;
- encoded input fade `E((1-t)x+tR_90x)`;
- encoded physical rotation `E(R_(90t)x)`;
- prescribed spatial action `rho(90t)E(x)`;
- unpenalized decoder geodesic;
- regularized geodesics from Stage 9;
- posterior Wasserstein/Fisher geodesics;
- learned shared-action orbit.

Measure:

- path length and energy under every declared metric;
- constant-speed error and Euclidean/covariant acceleration;
- output rotation angle, monotonicity, and winding;
- content feature drift, patch retrieval, and distance preservation;
- output sharpness, gradients, overshoot, SSIM/MAE to the physical-rotation
  reference, and difference maps;
- distance to represented states/orbit tube;
- latent norm, posterior plausibility, and near-null-direction usage;
- sensitivity to knots, initialization, optimizer, and direction reversal.

An orbit of isometries is not automatically a geodesic of the full manifold.
Report whether the free geodesic rotates, fades, deletes/recreates structure,
or uses another shortcut. A rotation-faithful claim requires the constrained
path to improve over the line on evaluation WSIs without receiving the target
rotation frames as a penalty.

Use each optimized quarter path `gamma_0` to form a candidate closed loop

`gamma_k(t)=rho(r)^k gamma_0(t)`, `k=0,1,2,3`.

Test endpoint continuity, tangent continuity, decoded closure, and composition.

## Stage 11: Cross-Content Interpolation

For each predetermined pair `(x1,x2)`, compare:

1. input line `x_t=(1-t)x1+t x2`;
2. encoded input line `E(x_t)`;
3. latent line `(1-t)E(x1)+tE(x2)`;
4. `C4`-aligned latent line to `rho(g*)E(x2)`;
5. decoder geodesic with `lambda=0`;
6. regularized decoder geodesics;
7. posterior Wasserstein/Fisher geodesics;
8. quotient geodesic with a fixed display gauge or parallel-transported phase.

The midpoint/Jensen defect is

`Delta_E(t)=E_mu((1-t)x1+t x2)-[(1-t)E_mu(x1)+tE_mu(x2)]`.

Measure this mean-field vector under Euclidean, functional, and local decoder
metrics. Do not apply Wasserstein/Fisher directly to it. Instead define the two
posterior paths explicitly: the encoded distribution `q(x_t)` and the analytic
diagonal-W2 or validated Fisher interpolation between `q(x1),q(x2)`, then
measure their distributional distance. Decode all paths after both common
parameter `t` and arc-length reparameterization.

The input line is the Euclidean image-space geodesic and commonly appears as a
fade. Recovering it from a decoder geodesic reveals geometry but is not by
itself more semantic. There is no ground-truth content interpolation, so this
stage remains qualitative/descriptive and may not decide model superiority.

## Stage 12: PGA In Three Geometries

PGA begins only after the particular geometry supplies a validated tangent
chart, Exp/Log construction, conditioning, optimizer convergence, and
rank-0/12 checks. A generic metric space does not automatically provide these.

For data `p_i` on a validated Riemannian manifold/chart, compute a
Fréchet/Karcher mean

`p_bar=argmin_p sum_i d_g(p,p_i)^2`.

Estimate log maps as initial velocities of stable geodesics

`v_i=Log_(p_bar)(p_i) in T_(p_bar)M`,

perform metric PCA in the tangent space, and map principal directions back by

`gamma_j(t)=Exp_(p_bar)(t v_j)`.

Run three explicitly separate analyses. Exact diagonal-W2 PGA is the mandatory
all-25 analytic route. Decoder-chart PGA first runs the ranks-0/12 numerical
pilot and expands to the WSI-disjoint evaluation set only through its bounded
amendment; an all-25 decoder-PGA remains a later bounded amendment. Exact-C4
quotient PGA uses the 25 orbit classes only when its regular-stratum gate passes.
Thus all three are attempted, but a failed mathematical or numerical gate is a
scientific limitation rather than permission to manufacture a log map.

### 12A. Mean-field decoder PGA

Use the explicit decoder chart `S_U` with primary `d=32` and fixed
`d=8,16,64` sensitivities, only while `G_U` remains uniformly positive
definite on every accepted path. Run orbit-only PGA and pooled
content-plus-pose PGA. These are PGAs of the reduced decoder chart, not of the
unknown full `M_mu`. Report endpoint projection and omitted full-field and
decoder-visible energy. If the gate fails, report local decoder-metric tangent
PCA instead and mark decoder PGA unresolved.

### 12B. Posterior-distribution PGA

Use `(mu,sigma)` under the flat exact diagonal-Wasserstein chart and, separately,
the explicit positive-sigma Fisher product manifold/chart with locked sigma
floor. Do not pool the metrics. Exact-C4 states are primary; continuous
re-diagonalized states are only sensitivity. Compare principal geodesics of
mean variation, uncertainty variation, and their coupling.

### 12C. Exact-C4 quotient PGA

Call this **exact-C4 quotient PGA**, not content PGA. It proceeds only on a
regular stratum with a metric that passed representative-invariance and a
unique alignment branch separated by the locked margin. Align by exact `C4`
and use

`d_Q([z_i],[z_j])=min_g d(z_i,rho(g)z_j)`.

Ambiguous branches are preserved as separate sensitivity analyses; they cannot
be collapsed into one log map. Visualize principal quotient geodesics under a
declared local gauge. Exact `C4` removes quarter-turn identity but not a
continuous pose tangent. Use the term *content PGA* only after H5 and H7 pass
and a continuous horizontal chart is independently validated.

For every PGA report:

- convergence and initialization dependence of the mean;
- distance of observations from the normal neighborhood;
- uniqueness/ambiguity of projections;
- tangent approximation error;
- principal geodesic versus linear-component reconstruction;
- decoded traversals at fixed geodesic distances;
- stability across fit-WSI resamples and retained dimension;
- normal versus `SO(2)` under identical settings.

PGA is descriptive with `n=25`. The 100 `C4` states are repeated states of 25
contents, not 100 independent samples.

## Stage 13: Discrete Connection, Gauge, And Vector Diffusion

Construct content-neighbor graphs on the 25 orbit classes using separately:

- Euclidean field exact-C4 quotient distance;
- decoder exact-C4 quotient distance only if its action-isometry gate passes,
  otherwise the labeled symmetric orbit-set dissimilarity;
- exact-C4 posterior Wasserstein quotient distance.

Primary graph degree is fixed at `k=4`; `k=3,5,6` are sensitivity analyses.
The primary adjacency is built once from the model-pooled Euclidean input-field
quotient distance so both models are compared on identical edges. Model-specific
latent graphs are descriptive within-model objects and their raw spectral gaps
are not ordered as if they shared a scale.
For each edge, retain every exact `C4` minimizer, its margin over the next best
alignment, and a weak-orientation flag. Do not force a unique phase when the
alignment is ambiguous.

The connection matrix uses only edges with a unique minimizer above the locked
margin. Ambiguous edges remain as labeled multi-edges in alignment and
sensitivity artifacts but are excluded from the orthogonal connection block.
If that fixed rule disconnects the primary graph, H8 is unresolved; averaging
several group matrices into a non-orthogonal pseudo-connection is forbidden.

Build:

- scalar/group synchronization for `C4` phase;
- block connection Laplacian using the relevant real group representation;
- vector diffusion coordinates/distances;
- cycle/holonomy residuals on all graph triangles;
- aligned local tangent/projector transports by Procrustes;
- a continuous mechanical connection only if H5 passes and the continuous
  action, chart, metric-isometry, positive-definiteness, and vertical-norm gates
  all pass.

Compare graph neighborhood preservation, patch retrieval, alignment margins,
cycle consistency, spectral gaps, and sensitivity to metric/degree. Relate
failures to input self-symmetry, low Lie-tangent norm, low `m=1` amplitude, and
posterior uncertainty.

With 25 nodes this is one exploratory fixed-set local-frame visualization.
Edges and triangles are not samples. Global spectra/cycle fractions are
reported without patch-level confidence intervals, with complete
leave-one-WSI-out refits as sensitivity; any cluster resample must rebuild the
entire basis, graph, alignment, and statistic. Do not make spectral-convergence,
topology, population, or global-bundle claims.

## Stage 14: Parallel Transport

Use three levels, each labeled separately.

1. **Group transport:** `xi -> rho(g)xi`, exact for lattice `C4`; the continuum
   formula is prescribed for `SO(2)`, while a sampled non-cardinal operator is
   labeled approximate and carries its interpolation floor.
2. **Discrete empirical transport:** align local functional/metric principal
   frames along graph edges by Procrustes; evaluate cycle consistency through
   the connection Laplacian.
3. **Reduced Levi-Civita approximation:** along stable decoder/posterior
   geodesics, transport vectors using a validated reduced metric. The
   result-blind preflight compares a covariant ODE, Schild's ladder, and pole
   ladder on Euclidean, sphere, and hyperbolic fixtures; the stage amendment
   locks one method, discretization, and call cap before model output. If
   first/second-order autodiff or the analytic error gate fails, this level is
   an accepted unresolved result rather than replaced after inspection.

Never materialize full Christoffel tensors. Compare transported orbit tangents,
content tangents, and principal directions across patches. Report norm
preservation, forward/backward closure, path dependence, and holonomy. A
transported semantic analogy is a qualitative illustration unless it transfers
to evaluation WSIs under a prespecified metric.

## Stage 15: Exact D4 Extension

Repeat the interpolation-free field, decoder, posterior, covariance, quotient,
and connection analysis for the eight exact `D4` elements. Decompose using
real `D4` irreducible sectors rather than reusing `SO(2)` frequency labels.

Require:

- the full multiplication table and inverse relations;
- compatibility of one reflection with all generated reflections;
- per-transform and joint decoder commutation;
- posterior/functional sector transformation;
- evaluation-WSI transfer;
- cycle consistency in the `D4` connection graph.

For final F0 fields, spatial reflection with even-scalar channel parity is a
prespecified extension/control, not an architectural consequence of `SO(2)`;
where identifiable, the odd-pseudoscalar extension is a mandatory sensitivity.
Odd/even sign parity applies to posterior means or internal feature fields;
`sigma` and `logvar` transform only by coordinate permutation because a sign
flip does not change covariance. For hidden F1 fields, vector and pseudovector
reflection conventions are separate hypotheses. The current single
`flip_diag` success is not a `D4` result unless this complete gate passes.

## Metrics, Sampling, And Statistical Contract

- Classify every endpoint as patch-level, evaluation-WSI-level, or a
  one-realization global fitted object before execution.
- Reduce every repeated angle/channel/mode/Monte Carlo quantity within patch,
  then reduce patches within WSI before biological model comparison. Report all
  25 patch rows, all 16 WSI rows, median, IQR, favorable count, and the fixed
  margin.
- Fitted transfer reports all eight evaluation-patch rows and five WSI
  reductions. Fit rows are diagnostics only. The set is already observed and
  is never described as independent confirmation.
- Descriptive WSI-cluster bootstrap intervals are role-stratified: resample and
  refit complete learned objects within the 11 fit WSIs, then evaluate on a
  separately resampled set of the five evaluation WSIs. They are
  fixed-validation stability intervals, not population confidence intervals.
- Use 10,000 paired cluster draws with replacement, preserving every patch of a
  sampled WSI and its multiplicity. Fit and evaluation roles are resampled
  independently without crossing roles. The seed is derived from
  `identity_root_sha` under purpose `paired_wsi_bootstrap`; report the 2.5th and
  97.5th empirical percentiles of the WSI-median paired effect. Global fitted
  objects are recomputed inside each draw or receive no bootstrap interval.
- Covariance commutators/spectra, shared fitted operators, graph spectra, and
  cycle summaries are global objects. Do not turn their modes, edges, cycles,
  or leave-one-out versions into pseudo-replicates; report one fixed-set value
  plus complete leave-one-WSI-out refits.
- Resample paired patches or WSI reductions, never individual angles, modes,
  channels, knots, graph edges, or posterior samples.
- Multiple metrics answer different geometric questions. Do not select one as
  primary after seeing model order. Report contradictions explicitly.
- Record denominator/nondegeneracy signals for every normalized metric.
- Angle prediction uses circular loss and reports winding; content metrics use
  patch retrieval, between-patch distance stress, and within/between variance.
- Cross-model comparisons use common physical output units or within-model
  normalization against the same no-action/known-action control. Learned
  families have matched capacity, initialization, operator-call budget, and
  convergence rules. F1-only results are mechanistic, not comparative.
- Every optimization reports all starts, convergence, final objective terms,
  failed starts, and result-independent stopping rules.

## Visualization Contract

Produce compact advisor-readable figures, with normal and `SO(2)` on matched
scales where coordinates are comparable:

1. native versus padded support, angle-zero identity, and boundary-ring maps;
2. exact `C4` commute square from input to encoded/action fields to decoder;
3. paired maps of `O_enc` and `O_act`, connected at equal angles and colored by
   Euclidean, decoder, Wasserstein, and Fisher distance;
4. exact `C4` sector-energy and covariance-commutator summary;
5. functional/steerable PCA singular spectra within every sector;
6. `m x multiplicity` harmonic-energy heatmaps for the continuous path;
7. cumulative Euclidean, `L2/H1`, decoder-visible, Wasserstein, and Fisher rank;
8. complex `q=+/-1` and continuous `m=1` planes with phase/amplitude;
9. modal field/reconstruction ladders: invariant, first rotational sector,
   increasing harmonic cutoff, complete field;
10. Lie-generator field maps, `A_Zz`, `dE(A_Xx)`, `J_D(A_Zz)`, and residuals;
11. decoder Jacobian singular spectra and sensitive/near-null spatial modes;
12. local decoder-metric ellipses in selected two-dimensional tangent planes;
13. posterior `mu/sigma` orbit and uncertainty maps under Wasserstein/Fisher;
14. shared-action evaluation prediction versus dimension/frequency and horizon;
15. decoder-preimage solutions from every initialization and penalty level;
16. known-rotation path strips comparing input fade, latent line, encoded
    rotation, spatial action, geodesics, and learned action;
17. path energy, speed, acceleration, rotation angle, content drift, and
    off-manifold distance;
18. cross-content interpolation strips for the four fixed pairs;
19. Fréchet means and decoded principal geodesic traversals for all three PGAs;
20. quotient graph with `C4` rosettes, alignment margins, and ambiguous nodes;
21. connection-Laplacian/vector-diffusion embedding and cycle-holonomy map;
22. transported tangent/principal-direction analogies with norm/closure error;
23. exact `D4` Cayley/group table with per-element errors and irreducible-sector
    energy;
24. one integrated evidence ladder separating architecture, exact empirical
    verification, learned transfer, local geometric interpretation, and
    speculation.

Every orbit plot uses isotropic axes. Separate model PCA bases, signs, colors,
or nonlinear embeddings are never placed on common numeric axes as if aligned.
All selected examples are ranks 0/12 or the fixed cross-content pairs.

## Randomness And Tuning Budget

- Canonicalize a projection of the machine contract that omits derived seeds,
  derived work IDs, output hashes, receipts, and runtime fields. Its SHA-256 is
  `identity_root_sha`.
  Derive each seed as the fixed-width integer digest of a length-delimited,
  domain-separated tuple `(schema_version,identity_root_sha,"seed",purpose)`.
  Derive work IDs from the same root under the separate `"work"` domain. Insert
  seeds and work IDs, then compute the final contract SHA. Shard headers and
  manifests bind that final SHA, but it is not an input to derived identities.
  This avoids self-referential hashes.
- Randomized SVD uses at least three prespecified seeds for stability, not for
  selecting the best result.
- Geodesic/action optimizations use a fixed initialization set and report all
  starts. Select by the detached CPU-FP64 recomputation of the complete
  objective subject to endpoint/convergence gates. Values within
  `max(1e-10,1e-6*abs(best))` are ties and all tied candidates remain in the
  artifact; this rule is fixed before remote output.
- Hyperparameters are selected from analytic fixtures and fit-patch numerical
  stability only. Held-out patches and normal-versus-`SO(2)` ordering are never
  used.
- Dense angle resolution, basis cutoffs, graph degree, PGA dimension, knot
  count, and regularization ladder are fixed above or finalized by the
  result-blind Stage 0/1 gates before any unmasked scientific comparison.

## Precision And Numerical Acceptance

The final machine contract must provide explicit tolerances for:

- exact group laws and hand-indexed array actions;
- basis/projector idempotence and orthogonality;
- Fourier-Bessel roundtrip and quadrature conditioning;
- eager/compiled and CPU/T4 parity;
- JVP finite-difference and adjoint consistency;
- randomized spectral residual/captured-energy certificates;
- restricted metric symmetry and positive definiteness;
- geodesic endpoint, reversal, refinement, and reparameterization consistency;
- Fréchet mean stationarity and initialization sensitivity;
- transport norm/roundtrip error;
- connection-graph gauge covariance;
- continuation/resume numerical tolerance and exact work/config/checkpoint
  identity.

Model forwards, JVP/VJP, and differentiable path objectives remain FP32. Once
detached, final norms, traces, certificate statistics, metric-isometry
residuals, and accepted path energies accumulate in CPU FP64. Report both the
FP32 differentiable objective and FP64 recomputation for accepted paths.

If a reduced metric has condition number above the contract limit or a smallest
retained eigenvalue below its relative floor, call it practically degenerate.
Report an unregularized pseudometric result, then use `G+lambda I` only as an
explicit regularized sensitivity. Do not silently clip eigenvalues.

## Multi-Session Kaggle And Resume Contract

### Immutable work units

Every expensive task is identified from `identity_root_sha` by SHA-256 of
canonical length-delimited JSON containing at least `schema_version`, stage, model,
WSI, patch, domain/support, precision, backend/compiler mode, initialization,
refinement, angle block, method, every scientific/numerical parameter, and
config SHA. Concatenated positional strings are forbidden because distinct
work units can otherwise collide.

Examples include one patch/model continuous block, one Jacobian probe block,
one geodesic path/start/refinement, one Monte Carlo block, or one PGA log-map.
The locked contract enumerates every expected `work_id` before launch.

### Per-session package

Every session writes a new immutable output root containing:

- `run_contract.json` and its SHA;
- `predecessor_manifest.json` when continuing;
- `runtime.json` with hardware/software/precision/compile/determinism identity;
- `work_units.jsonl` with expected, completed, failed, and pending status;
- `shards/<work_id>.*` with self-describing arrays and hashes;
- `checkpoints/<work_id>.*` for resumable optimizer/path state where needed;
- `metrics_partial.json` containing no final scientific decision;
- `manifest.json` hashing every accepted output;
- `status.json` marked `partial`, `complete`, or `failed`.

Publication uses a temporary output root, flush/fsync of payloads and directory
metadata, manifest creation that excludes the manifest itself, and an atomic
rename. A work unit is complete only after
its payload hash, schema, finite checks, and manifest row are durable. A
continuation runs under a new unique kernel slug, validates the entire mounted
predecessor manifest and expected manifest SHA, copies no result into a
different identity, skips only hash-valid completed units, and processes the
exact pending set. It records the full immutable ancestor DAG rather than a
mutable same-slug reference.

### Wall-time behavior

- Reserve at least 90 minutes from the observed Kaggle session limit for
  packaging and validation.
- Before starting a new work unit, project its cost from completed units and
  stop cleanly if it cannot finish within the reserve.
- Long geodesic/PGA units checkpoint at deterministic iteration boundaries;
  checkpoint includes knots/mean, optimizer state, Python RNG, NumPy RNG, CPU
  Torch RNG, every CUDA-generator state, sampler/work-order state, deterministic
  unit order, iteration, objective history, and source/config hashes.
- Resuming an interrupted optimization must reproduce a local synthetic
  uninterrupted reference within the numeric tolerance.
- A nonfinite or scientific convergence failure is a completed negative work
  unit, not grounds for an automatic retry with new settings.
- A platform failure may continue from the last valid checkpoint but may not
  change method, seed, precision, or tolerances without amendment.

### Stage plan

The dependency roadmap is:

1. `A`: exact `C4/D4`, padding, and continuous-10-degree forward fields;
2. `B`: functional/harmonic decompositions and posterior analytic geometry;
3. `C`: Lie JVPs and randomized/restricted decoder metric;
4. `D`: decoder-preimage diagnostics and ranks-0/12 geodesic/PGA pilot;
5. `E`: eligible evaluation-WSI geodesic, posterior, and shared-action units;
6. `F`: quotient/PGA/transport/connection synthesis and final figures.

Each letter becomes its own bounded amendment only after all predecessors and
numeric gates are known. Each amendment defaults to one launch plus one
continuation, lists exact operator calls and mandatory work units, and needs
its exact push authorization. No stage may silently consume unlimited versions.
If projection exceeds that ceiling, split future work into another amendment
before seeing the affected model comparison; do not drop methods or expand
successful branches after results.

### Storage policy

Do not persist full Jacobians, all decoded frames, or duplicated complete
latent sweeps. Stream activations and save:

- exact source fields needed for audit;
- coefficients, bases, reduced Gram matrices, spectra, and residual
  certificates;
- selected ranks 0/12 and fixed-pair path arrays/frames;
- per-patch scalar rows;
- optimizer state only for pending paths;
- enough data to deterministically rebuild every figure.

The complete two-model one-degree native `mu` sweep is about 1.18 GB FP32;
native `mu+logvar` is 2.36 GB. The `48x48` padded equivalents are about 2.65 GB
and 5.31 GB. An all-angle padded decoded sweep would be about 31.85 GB and is
forbidden. A padded dense Jacobian would be about 65.2 GB FP32 and is forbidden,
as is a full Jacobian at every other size. Every stage amendment records peak
VRAM, session-output, predecessor-mount, and local-download ceilings and names
the path/optimizer arrays that persist. Reuse hash-verified shards instead of
writing decoded movies for all angles.

## Expected Compute And Stop Rules

Initial T4 planning ranges, to be replaced by Stage 0 measurements:

| Work | Expected order |
| --- | ---: |
| Exact `C4`, functional sectors, analytic posterior metrics | minutes |
| Padded 10-degree paired sweep | under one hour |
| Padded one-degree paired sweep | roughly one hour |
| Restricted/randomized decoder spectra | one to four hours |
| Selected rank-0/12 action/geodesic pilot | one to three hours |
| Eight-patch/five-WSI geodesic/action work | several hours per locked configuration family |
| Reduced PGA and log maps | potentially several hours to days for decoder geometry; analytic W2 is cheap |
| Connection Laplacian/vector diffusion after distances | minutes |

The count of source patches is small, but a 64-knot path optimized for hundreds
of iterations creates tens of thousands of decoder forward/backward evaluations
per path. `torch.compile` may accelerate stable repeated shapes but does not
remove this algorithmic multiplication. Microbatch knots/probes rather than
lowering scientific precision after an OOM.

Stop a stage and report it unresolved when:

- the required metric is practically degenerate after the declared reduction;
- geodesics fail endpoint/convergence/refinement checks for most patches;
- padding fails identity and no primary-eligible continuous domain remains;
- randomized residual certificates fail within the locked operator-call
  budget;
- continuation state cannot be authenticated exactly;
- result completion would require sealed data, retraining, an unplanned remote
  dataset write, or a changed scientific rule.

### Conditional stage DAG

Numerically valid scientific failures are complete negative results. The
artifact contract marks downstream items explicitly:

| Gate | Pass | Scientific fail | Numerical/identifiability fail |
| --- | --- | --- | --- |
| Padding identity/interpolation | run padded continuous pilot | keep exact C4 and native/disk controls; padded claims not applicable | continuous padded action unresolved |
| Continuous pilot | run eligible dense/shared-action stage | H5 not supported; continuous geodesic/action figures not applicable | H5 unresolved |
| Restricted decoder metric | run decoder-chart paths/PGA | H3/H6 not supported where eligible | decoder Riemannian geometry unresolved; retain decoder-energy paths |
| H5 shared action | permit continuous quotient/connection analysis | no continuous connection/content quotient; exact C4 remains | continuous bundle questions unresolved |
| Quotient isometry/alignment | run exact-C4 quotient PGA/connection | use only alignment/orbit-set diagnostics | quotient geometry unresolved |
| First/second-order transport | run locked reduced transport | transport not supported | Levi-Civita transport unresolved |

The final manifest lists every planned output as `required`, `not_applicable`,
or `unresolved` with the controlling gate. It never substitutes an easier
method after failure.

## Outputs And Acceptance Artifacts

Accepted local root, created only after complete receipt-bound download and
validation:

`runs/local/functional_riemannian_geometry_v1/`.

Required compact artifacts:

1. `functional_riemannian_summary.json` with all formulas, per-patch values,
   paired comparisons, fit/evaluation results, numerical gates, and conclusions;
2. `run_ledger.json` linking every Kaggle session, continuation, contract,
   predecessor, code, checkpoint, source, runtime, and output hash;
3. `bases_and_spectra.npz` with exact projectors, functional bases, reduced
   spectra, and randomized residual certificates;
4. `selected_paths.npz` with all fixed selected paths, starts, metrics, and
   enough frames to regenerate path figures;
5. `posterior_geometry.npz` with Wasserstein/Fisher reduced results;
6. `graph_and_transport.npz` with quotient alignments, ambiguity margins,
   connection matrices, spectra, cycles, and transports;
7. every gate-applicable compact figure in the 24-item visualization contract,
   combined where legibility permits, plus explicit placeholder/status rows for
   items made not applicable or unresolved by the conditional DAG;
8. `manifest.json` hashing every artifact and naming Specs 0050/0051 evidence
   as predecessors without modifying it;
9. an independent-review addendum for mathematics, randomized numerics,
   statistics, scientific claims, resumability/provenance, and visual layout.
10. `claim_ledger.json` mapping each sentence-level scientific conclusion to
    hypothesis, primary endpoint, eligible population/domain, gate status,
    artifact/work-unit hashes, and exact predecessor Specs 0050/0051 hashes.

## Acceptance Criteria

1. Only the fixed checkpoints and ordered validation 25 are used; no parameter
   update, sealed access, result-dependent example, or source mutation occurs.
2. Exact `C4` field/group tests, scalar/F1 fixtures, Lie signs, posterior
   formulas, functional bases, and quotient gauge covariance pass independent
   analytic tests.
3. Padding/native/expanded domains and interpolation/aliasing floors are
   reported separately; no continuous result is generalized from a failed
   identity gate.
4. Functional PCA/SVD distinguishes exact group sectors, continuous modes,
   multiplicity rank, retained raw energy, and decoder-visible energy on held-
   out patches.
5. JVP/VJP and randomized algorithms provide residual certificates and stable
   subspaces without materializing native full Jacobians/metrics.
6. Decoder metric rank/conditioning is established before Riemannian language;
   regularized and unregularized results remain separate.
7. Lie-derivative residuals, two-orbit distances, metric invariance, posterior
   geometry, shared-action transfer, decoder-preimage nonuniqueness, and every
   gate-applicable declared control are complete for both models.
8. Unpenalized geodesics are reported before regularized paths. All starts,
   failed convergence, shortcut behavior, and refinement sensitivity remain
   visible.
9. PGA is attempted and adjudicated separately for the reduced mean-field
   decoder chart, exact posterior-distribution geometry, and regular exact-C4
   quotient stratum. Each is completed when eligible or carries its exact
   numerical/mathematical blocker; no generic metric-space PGA is claimed.
10. Connection/transport results preserve gauge covariance, ambiguity, and
    stabilizers; no topology/global-bundle claim is made.
11. `D4` is claimed only if its complete group gate passes; otherwise report
    transform-specific robustness.
12. Every remote stage is receipt-bound, resumable, immutable, and reproduced
    in the final run ledger; incomplete work cannot enter final decisions.
13. Every biological population comparison uses WSI reductions, persists all
   patch/WSI rows, classifies global fitted objects separately, and labels
   intervals descriptive fixed-validation evidence.
14. Figures are reproducible from compact arrays, use honest matched scales,
    fixed examples, and clearly separate architecture, empirical verification,
    transfer, interpretation, and speculation.
15. Independent clean-context reviewers report no unresolved P0/P1 in
    mathematics, numerical linear algebra, statistics, workflow/provenance,
    scientific interpretation, or visualization.
16. Focused tests, Ruff, formatting, BasedPyright, the repository quality gate,
    `git diff --check`, repository preflight, and workspace preflight pass, or
    unrelated existing failures are identified exactly without weakening the
    new gate.

## Planned Code And Contract Surface

Exact filenames may be narrowed during lock review, but responsibilities must
remain separated:

- `src/eqvae/evaluation/functional_geometry.py`: supports, group projectors,
  functional bases, regularity, harmonic/multiplicity analysis;
- `src/eqvae/evaluation/riemannian_geometry.py`: operator metrics, randomized
  spectra, geodesics, Fréchet/PGA, and transport;
- `src/eqvae/evaluation/posterior_geometry.py`: Wasserstein/Fisher posterior
  operations;
- `src/eqvae/evaluation/quotient_geometry.py`: alignments, gauge, connection
  Laplacian, and vector diffusion;
- `src/eqvae/cli/run_functional_riemannian_geometry.py`: local/stage runner and
  immutable output schemas;
- `kaggle/kernels/functional_riemannian_geometry/`: guarded reusable private
  stage template;
- `docs/data/spec0053_*_contract.json`: immutable master and per-stage machine
  contracts created before the corresponding launch;
- focused tests for every module, kernel guard, resume state, mutant, and
  mathematical fixture.

No optional dependency is added until the implementation review proves that
the required operations cannot be implemented with the locked PyTorch/NumPy/
SciPy stack. Dependency additions require `pyproject.toml`/`uv.lock` updates and
separate permission before syncing.

## Verification Plan

Before lock:

- independent mathematical/functional review;
- independent randomized-numerics/Riemannian review;
- independent statistics/resume/provenance review;
- resolve every P0/P1 and every lock-blocking P2.

Before each remote stage:

```bash
.venv/bin/pytest -q <focused-spec0053-tests>
.venv/bin/ruff check <touched-spec0053-python>
.venv/bin/ruff format --check <touched-spec0053-python>
.venv/bin/basedpyright <touched-spec0053-python>
./scripts/kaggle_kernel.sh build kaggle/kernels/functional_riemannian_geometry
./scripts/kaggle_kernel.sh validate kaggle/kernels/functional_riemannian_geometry
./scripts/kaggle_kernel.sh check kaggle/kernels/functional_riemannian_geometry
./scripts/agent_preflight.sh
git diff --check
```

Before final acceptance:

```bash
./scripts/python_quality.sh
./scripts/agent_preflight.sh
../agent_preflight.sh
git diff --check
```

Render and inspect every final figure at original resolution. Independently
recompute a sample of per-patch rows, basis energies, JVP adjoints, geodesic
energies, posterior distances, graph alignments, and continuation hashes.

## Adversarial Design Review Record

On 2026-09-10 three independent clean-context reviews examined the draft from
mathematical, numerical-linear-algebra/resume, and statistical-design
perspectives. Initial findings exposed the non-isometric quotient issue,
continuous diagonal-covariance error, undefined manifold domain for geodesics,
C4-to-SO2 alias ambiguity, WSI leakage, target-informed action/geodesic
evidence, pseudo-replication, FP16/Jacobian conditioning, randomized-certificate
coverage, and self-referential resume identities. The plan above incorporates
those corrections. Final focused re-reviews reported no unresolved P0/P1
design finding in those scopes.

This review accepts the roadmap, not an implementation or launch. The spec
remains `draft active` until the result-blind preflight, exact tolerances,
continuation proof, bounded stage amendments, and machine contracts below are
complete and independently re-reviewed.

## Implementation Blockers Before Lock

1. Create and review the separately bounded result-blind preflight amendment:
   exactly one unique-slug launch plus at most one continuation and no
   model-comparison output.
2. Convert its benchmarks into exact per-amendment memory/time/operator-call,
   knot, iteration, microbatch, output/download, and session ceilings without
   viewing unmasked scientific comparisons.
3. Lock analytic tolerances for Lie derivatives, randomized residual/trace
   confidence certificates, reduced-rank noise floors, geodesics, every PGA,
   transport, quotient isometry, alignment ambiguity, and gauge covariance.
4. Validate `384/48` first and characterize GroupNorm/FixedF01FieldNorm size
   dependence; test `512/64` only on a passing common route. Preserve native
   square/disk as the result-independent fallback.
5. Prove exact private predecessor attachment with immutable
   owner/slug/version, embedded manifest SHA, unique child slug, complete
   ancestor DAG, older-shard access, fan-in, and final local aggregation.
6. Decide the primary antialiased continuous interpolation operator from
   analytic/input-only tests; retain corrected bilinear as control.
7. Fix the positive `H1` quadrature scale, Fourier-Bessel radial truncation,
   Fisher sigma/saturation floors, decoder-chart rank floor, condition-number
   ceiling, optimizer/stopping/call rules, and regularization weights from
   analytic and model-label-masked fit-only numerical scales.
8. Serialize the primary decision matrix, WSI split, C4 alias branch set,
   method priority, matched family capacities, conditional DAG, and all
   undefined-case rules before Stage A.
9. Produce immutable master/stage JSON contracts, collision-safe work IDs,
   identity-root projection, and unique guard phrases.

No scientific implementation or remote push begins while these blockers remain.

## Known Risks And Adversarial Checks

- A complete orthonormal basis change cannot create PCA rank reduction. Review
  must distinguish group-sector sparsity, multiplicity rank, intrinsic orbit
  dimension, linear span, and decoder-visible rank.
- C4 aliases `m` modulo four. Mutant tests must reject code that labels a C4
  sector as a unique continuous harmonic.
- Symmetrized covariance commutes by construction. The raw commutator and held-
  out performance must remain visible.
- Padding can change GroupNorm globally. Angle-zero and exact-C4 padded/native
  controls must precede continuous interpretation.
- Interpolation can manufacture smoothness; compare analytic floors, input
  paths, operators, support variants, and angular resolutions.
- `J_D^T J_D` can be ill conditioned. Reviewers must test near-null modes,
  FP16 underflow, unstable power iteration, false convergence, and hidden
  eigenvalue clipping.
- Randomized algorithms can look stable while missing a flat spectral tail.
  Require independent trace estimates and residual certificates.
- Decoder geodesics can fade, blur, leave the represented set, or exploit
  insensitive directions. These are results, not reasons to hide the
  unpenalized path.
- A regularizer can encode the desired answer. Every term and unregularized
  baseline must be displayed separately.
- Fréchet means, log maps, and projections can be nonunique near cut loci or
  stabilizers. Preserve multiple minimizers and initialization sensitivity.
- Reduced PGA can manufacture low dimension. Report omitted raw/visible energy
  and stability across `d`.
- Graph alignments can be dominated by content similarity or arbitrary nearest
  neighbors. Report margins, graph-degree sensitivity, and all cycles.
- Gauge fixing can hide phase inconsistency. Verify gauge-covariant quantities
  under random node-wise `C4` changes.
- Exact decoder C4 action does not imply encoder equivariance, continuous
  `SO(2)`, D4, factorization, or a global product.
- The models have separate channel coordinates. Never compare individual
  channel loadings or PCA axes across checkpoints without an independently
  fit/evaluation alignment.
- Twenty-five contents are insufficient for global dimension, topology,
  population, or causal claims.
- The large method family creates multiplicity and narrative-selection risk.
  Preserve the evidence hierarchy and primary questions; contradictions are
  first-class outputs.

## Geometric Interpretation Required In Every Final Synthesis

Report separately:

- what is guaranteed by field types and exact array actions;
- what exact `C4` behavior is empirically verified;
- what changes under native versus padded continuous domains;
- what is organized by group harmonics and functional bases;
- what is visible under decoder and posterior metrics;
- what the Lie-derivative tests verify or contradict;
- what shared action transfers across WSI-disjoint evaluation patches/angles;
- what unpenalized and constrained geodesics actually do;
- whether canonicalization preserves content;
- what connection/transport results support locally;
- which patches exhibit stabilizers or undefined phase;
- what remains speculative or impossible with the fixed 25.

## References And Related Files

Method references:

- Fletcher et al., *Principal Geodesic Analysis for the Study of Nonlinear
  Statistics of Shape*, IEEE TMI 2004, DOI `10.1109/TMI.2004.831793`.
- Halko, Martinsson, and Tropp, *Finding Structure with Randomness*,
  `https://arxiv.org/abs/0909.4061`.
- Arvanitidis, Hansen, and Hauberg, *Latent Space Oddity*,
  `https://arxiv.org/abs/1710.11379`.
- Shao, Kumar, and Fletcher, *The Riemannian Geometry of Deep Generative
  Models*, `https://arxiv.org/abs/1711.08014`.
- Arvanitidis et al., *Pulling Back Information Geometry*,
  `https://arxiv.org/abs/2106.05367`.
- Singer and Wu, *Vector Diffusion Maps and the Connection Laplacian*,
  `https://arxiv.org/abs/1102.0075`.
- *Riemannian Principal Component Analysis*, `arXiv:2506.00226`, reviewed but
  excluded from the primary plan because its UMAP-induced local distances do
  not provide the mechanistic decoder/posterior metric required here; RPCA also
  has no unique role beyond the three separately defined PGA geometries.

Repository contracts:

- `AGENTS.md`
- `CURRENT.md`
- `GOAL.md`
- `docs/specs/0038-frozen-vae-rotation-orbit-visualization.md`
- `docs/specs/0040-spatial-latent-pca-coherence-visualization.md`
- `docs/specs/0048-professor-final-experiment-report.md`
- `docs/specs/0050-corrected-rotation-geometry-validation.md`
- `docs/specs/0051-decoded-latent-transform-consistency.md`
- `docs/specs/0052-professor-report-decoded-transform-integration.md`
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`
- `docs/kaggle_cli_workflow.md`
- `docs/agentic_review_workflow.md`
