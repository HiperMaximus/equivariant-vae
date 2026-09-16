# Spec 0053: Functional And Riemannian Latent Geometry

Status: draft active; Stage A1 implemented and accepted; numerical calibration
v1/v2 completed `unresolved`; focused v3 is launch-ready; scientific Stage A2
remains blocked until a justified common numerical contract exists
Owner/workstream: frozen normal versus continuous-`SO(2)` VAE latent analysis
Last updated: 2026-09-16

## Purpose

Determine what rotations, content changes, and decoder-visible directions look
like in the latent fields of the two frozen VAEs. The central question is not
whether one flattened posterior mean has a small PCA rank. It is whether
rotation defines a coherent action, orbit, local quotient, or geodesic
structure after accounting for decoder redundancy.

The experiment may falsify every proposed structure. Exact `C4` is the
interpolation-free anchor; continuous `SO(2)` claims require separate evidence.

## Fixed Inputs And Scope

| Item | Value |
| --- | --- |
| Normal checkpoint | `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`; SHA-256 `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075` |
| `SO(2)` checkpoint | `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`; SHA-256 `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7` |
| Weight dataset | `maximshtefan/eqvae-vae-test-reconstruction-inputs-v1`, version 1; consumed files remain hash-verified |
| Patch dataset | `maximusshtefan/patches-pre-shuffled-ubc-ocean`, version 1; selected patch bytes remain hash-verified |
| Fixed patches | The accepted ordered validation 25; selector SHA-256 `ace244ecdd67aaa1ebc7d08065f1e4bfa3c0d54806d4f3b50fb38a3ae000447f` |
| Input | FP32 `3x256x256` normalized image |
| Latent state | Deterministic posterior mean `mu`, FP32 `16x32x32`; `logvar` is telemetry unless posterior geometry is explicitly studied |
| Exact group | `C4={e,r,r^2,r^3}`, `r=torch.rot90(+1)` |
| Primary pilot patches | Fixed ranks 0 and 12 |

Both checkpoints remain frozen. No training, sealed-test access, new example
selection, or model-dependent retuning is allowed.

The primary pilot is local and fits no cross-patch action or chart. Ranks 0 and
12 may use their own permitted endpoints for descriptive interpolation, but a
free continuation may use only that patch's `z_0`, `z_1`, and decoder
differentials evaluated without target access.

Any secondary cross-patch transfer analysis must exclude the complete pilot
WSIs `38349` and `59031`. Its pilot-safe fit pool is therefore the 15 ranks
`4,5,8,10,13,16,18,19,20,21,22,15,17,23,24` from ten WSIs. In particular,
rank 12 and the same-WSI rank 14 cannot contribute any coefficient, mean,
basis, threshold, dimension choice, initialization, or stopping decision used
to evaluate rank 12. “Fit” always means post-hoc analysis, never VAE training.

## Scientific Boundaries

- Four cardinal samples identify only exact `C4`, not a unique continuous
  generator or winding number.
- Four centered points span at most three dimensions by construction; their
  per-patch PCA rank is not evidence for a low-dimensional rotation manifold.
- A closed curve is not automatically smooth, an action orbit is not
  automatically geodesic, and a minimum-energy path is not automatically
  semantic.
- Decoder pullback geometry can be degenerate. Riemannian language is used only
  on a fixed stable positive-definite restricted immersion.
- A local chart is not evidence for a shared action. Failure of cross-patch
  transfer does not invalidate a patch-local geodesic.
- A curve forced through four anchors is a piecewise geodesic or Riemannian
  spline diagnostic, not evidence that one free geodesic generates the cycle.
- Parallel transport moves tangents, not points. Closed-loop tangent or frame
  mismatch is holonomy; torsion is not inferred from it.
- Results from ranks 0 and 12 are a numerical and visual pilot, not population
  evidence or a global manifold/bundle/topology claim.
- The two models have separate channel coordinates. Compare invariant
  quantities or independently aligned subspaces, not raw channel loadings.

## Geometry And Terminology

### Encoded and prescribed cycles

For a patch `x` and `k=0,1,2,3`, define

`e_k = E_mu(R_(90k)x)`

and

`a_k = rho(r)^k E_mu(x)`.

The encoded cycle `{e_k}` contains the representatives selected by the
encoder. The prescribed-action cycle `{a_k}` contains the orbit obtained by
acting directly on the original latent field. Neither may be substituted for
the other.

Strict encoder equivariance requires

`e_k = a_k`.

The experiment also tests the weaker relation

`[e_k]_D = [a_k]_D`,

meaning equivariance modulo decoder redundancy.

### Decoder fibers and pose orbits

A decoder fiber over output `y` is

`F_y={z:D(z)=y}`.

Only on a neighborhood where `D` has constant rank does the constant-rank
theorem identify the tangent of a regular fiber with
`T_z F_y=ker J_D(z)`. Without that condition, `ker J_D(z)` is only the instantaneous
null space. A numerically small but positive singular direction is an
`epsilon`-inactive direction, not an exact vertical or gauge direction. Two
distant latent states are called approximately decoder-equivalent here only
when a preregistered connected-bridge test passes; similar endpoint decodes
alone do not define a fiber or an equivalence relation.

A pose orbit `{rho(g)z}` is different: it normally moves across decoder
fibers over differently rotated outputs. Only in a separately established
content-versus-pose quotient may a rotation orbit be interpreted as a pose
fiber.

Operationally, corresponding encoded and prescribed representatives form

```text
e_k = E(R^k x)  -- approximate decoder-insensitive bridge --  a_k = rho^k E(x)
       | D                                               | D
       v                                                 v
    D(e_k)                ~= R^k D(e_0)               D(a_k)
```

Similar endpoint decodes are insufficient to establish one connected fiber.
The experiment also requires a low-decoder-energy bridge between the
representatives.

### Decoder pullback, restricted immersions, and conditional quotient

The decoder pullback form is

`g_z(u,v)=<J_D(z)u,J_D(z)v>`,

with `G(z)=J_D(z)^T J_D(z)`. It is positive semidefinite in the full latent
space; any exact instantaneous-null direction has zero length.

For a fixed orthonormal local chart `phi:Omega subset R^d -> Z`, let

`B(xi)=J_D(phi(xi)) J_phi(xi)`,

`G_phi(xi)=B(xi)^T B(xi)`.

Rank and conditioning are determined from the thin operator `B`, not from
squaring its condition number and guessing from noisy eigenvalues of `G_phi`.
Riemannian language is permitted only when `phi` is fixed and `C2` on a
declared trust region, `B` has constant full column rank there, and the result
is stable under chart, dimension, and knot refinement. Then `G_phi` is positive
definite and defines the Levi-Civita connection of that explicitly restricted
decoder immersion. A sequence of independently adapted frames does not by
itself define an atlas, connection, Exp, Log, or holonomy. If any chart criterion
fails, retain the computation but call it only a reduced decoder-energy path.

An exact quotient by decoder fibers additionally requires the local
constant-rank/fiber assumptions above plus
`dim(phi)=rank(J_D)` and transversality
`Im(J_phi) direct-sum ker(J_D)=T_z Z`. Stage A2 does not assume or expect a
small chart to pass those full-quotient criteria. It studies the Riemannian
geometry of an explicitly restricted immersion and separately tests
operational decoder equivalence with connected bridges. If positive singular
directions are discarded at a declared threshold, freeze a new lower-
dimensional chart before solving; never relabel the result as an exact decoder
quotient.

Latent paths may differ while their decoded curves agree. Such multiplicity is
possible gauge evidence and remains visible in the results.

### Point motion, tangent transport, and holonomy

For an admissible endpoint pair, a decoder-pullback geodesic is a locally
energy-minimizing curve

`gamma(0)=z_0`, `gamma(1)=z_1`,

with

`E(gamma)=integral ||J_D(gamma(t)) gamma_dot(t)||_X^2 dt`.

On a validated restricted immersion, `Log^phi_(z_0)(z_1_prime)` denotes the
initial chart-tangent velocity and `Exp^phi_(z_0)(t v_0)` its point
continuation. Decoding
`D(gamma(t))` produces the requested intermediate images. If all four anchors
are imposed, solve four sides independently and report the concatenation as a
teacher-forced piecewise path. A single free orbit claim instead requires the
first side alone to determine the later quarters.

Parallel transport propagates a tangent or frame along an already defined
curve. For a closed loop, compare the returned point, the transported rotation
tangent, and a transported chart-tangent frame with their initial values. Point
mismatch is a closure defect; only the tangent/frame return map around an
actually closed loop is holonomy. The primary smooth-closed-geodesic check is
that the rotation tangent returns to itself; a full chart frame may retain
nontrivial curvature-induced holonomy and is measured rather than required to
be the identity. On a validated stable chart the target connection is
Levi-Civita and therefore torsion-free by construction. The implemented
repeated-projection transport is only a discrete approximation and must
converge under knot refinement before supporting a holonomy claim.

### Exact-C4 Fourier sectors are a secondary diagnostic

Every four-tuple admits the orthonormal real DFT decomposition

`c_0=(z_0+z_1+z_2+z_3)/2`,

`c_2=(z_0-z_1+z_2-z_3)/2`,

`c_c=(z_0-z_2)/sqrt(2)`, and `c_s=(z_1-z_3)/sqrt(2)`.

These are respectively the `q=0`, `q=2`, and paired `q=+/-1` sectors of an
exact four-state sequence. `q=0` is invariant under a cyclic shift, `q=2`
changes sign, and `(c_c,c_s)` transforms as a real two-dimensional pair. This
is an algebraic identity for any four points, not evidence for an action,
geodesic, low-dimensional manifold, or continuous generator. The labels are
characters of `C4` modulo four; they do not identify a unique continuous
frequency or winding number and are unrelated to the architecture's F0/F1
field labels.

The DFT sectors may summarize a completed local cycle or support a separately
labelled cross-patch transfer test. They must not construct a primary local
geodesic chart, select its dimension, or expose withheld `z_2,z_3` to a free
continuation.

## Accepted Numerical Method

Decoder differential calculations use direct disposable JVP/VJP calls with
model parameters frozen:

`Gv=J_D(z)^T(J_D(z)v)`.

The accepted Stage A1 settings are:

- eager FP32 operator arithmetic;
- finite-difference JVP reference epsilon `0.008`;
- direct graphs released after every work unit;
- microbatch 4 for compatible independent directions;
- matrix-free SLQ with Lanczos depth `m=64`;
- `r=64` independent normalized Rademacher probes;
- FP64 tridiagonal eigensolves and 99% scalar intervals.

`m` controls Krylov depth per SLQ probe and `r` controls independent
replication. Neither is a latent projection dimension or a geodesic batch
size. A full Jacobian or full `G` is never materialized. Small reduced
`G_phi` matrices are computed exactly when validated reduced-geodesic or
transport calculations need them.

## Completed Stage A1

Stage A1 measured exact-`C4` action on all 25 fixed patches and matrix-free
decoder spectra at ranks 0 and 12 for all four cardinal angles.

The accepted scientific result is:

- The `SO(2)` decoder implements the prescribed exact-`C4` action at about
  `1e-6` RMS, versus normal-VAE medians `.165--.190`.
- The final posterior field is not strictly equivariant under the same simple
  spatial action; raw `mu/logvar` discrepancies are large and descriptively
  larger for `SO(2)`.
- Across the eight matched two-anchor states, the `SO(2)` pullback trace is
  `.895--.959` of normal and its estimated `d95/d99` are lower in every
  state.
- The spectral observations describe two anchors only. They establish neither
  a population effect nor a continuous orbit or geodesic.

The immutable Stage A1 implementation is preserved by Git history and Kaggle
kernel `maximshtefan/eqvae-fg-stage-a1-c4-28a08ab5/1`. Accepted output:
`runs/kaggle/functional_geometry_stage_a1_c4_slq_v1/functional_geometry_stage_a1_c4_slq_v1/stage_a1_result.json`,
SHA-256
`f59bf4eb7bba136c02cca24b237414da1dca23e4b94b781c4eb3d5978840e449`.

## Current Experiment: Exact-C4 Geodesics, Transport, And Fibers

### Pre-Stage A2 numerical calibration

Stage A2 begins with one private, non-scientific calibration probe. It uses
fixed ranks `4,8,10,16,24`, one per selected WSI, and excludes final-pilot
ranks `0,12` plus every patch from their WSIs `38349,59031`. The probe may
inspect each frozen decoder only to choose one common numerical contract from
the conservative joint envelope; it must not report a normal-versus-`SO(2)`
winner or contribute evidence to any Stage A2 claim.

The v1 candidate grid is preserved in its authenticated run artifact. It measured:

- nested affine charts `d=8,16,32`, built from the permitted endpoint secant
  plus seeded midpoint `G`-filtered Rademacher directions;
- thin-metric singular spectra at line parameters `0,1/2,1`, conditioning
  ratios, sampled two-sided local linearization radii using the secant plus
  four seeded full-support chart combinations per dimension, and finite-
  difference JVP scales;
- literal first-quarter path relaxation with `K=8`, Adam learning rates
  `.005,.02`, dimensions `8,16,32`, and milestones through 32 iterations on
  calibration ranks `4,16` for encoded and prescribed routes;
- one forward/backward runtime and memory measurement at `K=8,16,32,64`.

The selected contract must use the same chart dimensions, tolerance rules,
optimizer schedule, and ceilings for both models. The reducer is deterministic
and examines only worst-case values across all model/calibration workloads,
never a between-model difference. It selects the finite-difference epsilon with
minimum worst error among those at or below `.005`; the second-largest and
largest nested dimensions whose worst singular-value ratio is at least `1e-3`;
the largest sampled radius whose worst relative linearization error is at most
`.05`; the Adam rate with minimum worst final/initial energy ratio among
complete primary-and-refinement candidates that never touch the trust boundary
and reduce energy by at least 1%; and the earliest nonzero milestone within 1%
of every such candidate's best observed energy. Both selected dimensions must
pass the radius and optimizer criteria. `K=16` and `K=32` must both fit; `K=64` is
retained only as a feasible sensitivity setting. The diagnostic also reports
which dimensions/radii pass the other preregistered conditioning and linearity
targets, but does not relax the fixed selection criteria from observed results.

Only chart dimensions, finite-difference epsilon, sampled chart radius, Adam
rate/iterations and feasible path discretizations are adjustable here. Bridge,
closure/return, covariance/isometry and holonomy claim criteria remain outside the
calibration and model-specific numerical settings are prohibited. The
common reducer runs only after both model outputs are complete. The complete
artifact and decision are retained even if no common setting is viable. Raw
bytes, embeddings and decodes
for final-pilot ranks `0,12` and their WSIs are not re-accessed by this probe
and cannot enter its reducer; their already accepted Stage A1 diagnostics
therefore provide no calibration input. The full selector metadata is read only
to hash-verify the fixed source and resolve the allowlisted byte offsets.

This probe calibrates numerical scale and feasibility. It does not establish
a quotient, bridge, geodesic, continuous action, transport, or holonomy, and
its optimized paths are not Stage A2 results.

#### Calibration v1 outcome

Private version `maximshtefan/eqvae-fg-stage-a2-calibration/1` completed in
`2098.65` seconds and is authenticated by its local launch/download receipts.
Its common decision is `unresolved` (selection SHA-256
`a17542a87ba73f318505b45d2f73f27fd09fbf4787ca096f11421f5a02d527d1`)
for two independent reasons:

- no sampled transverse radius passed the fixed `.05` relative-linearity
  criterion; the worst error was `.34225` already at radius `.01`;
- batched `K=32` completed for normal but exhausted memory for `SO(2)`, while
  `K=64` exhausted memory for both. Only `K=8,16` completed jointly.

The successful components are diagnostic only: all candidate charts passed
conditioning, epsilon `.016` had worst central-FD error `.00217`, and Adam
`.005` reduced the path energy in all primary/refinement calibration units.
The output's partially populated `selected_numerics` is not a usable scientific
contract because the global status is `unresolved`. It must not unlock final
pilots or freeze `.016`, dimensions `16/32`, `.005/32` optimizer settings, or
`K=8/16` as Stage A2 parameters.

#### Calibration v2 contract

V2 is a patch of the executed v1 runner, not a replacement pipeline. It keeps
the same calibration examples, WSI exclusions, routes, frozen checkpoints,
seeds, two-T4 mapping and common worst-case reducer. Final-pilot ranks and WSIs
remain inaccessible.

The patch changes only the failed numerical axes. The finite-difference grid is
`.004,.008,.016,.032,.064`; local first-order radii are
`.00025,.0005,.001,.0015,.002,.003,.005`; and the v1-supported dimensions
`d=16,32` and learning rate `.005` are validated at `K=16,32`. Every fixed
`(model,rank,route,d,K)` candidate creates Adam once and runs continuously
through milestones `0,8,16,32,48,64,96,128,160,192,256,320,384`. The reducer
selects the earliest common nonzero milestone within 1% of each path's best
observed energy after at least 1% improvement. If only milestone 384 qualifies,
the iteration ceiling is unresolved rather than silently accepted.

Path energy is the unchanged sum over edges, evaluated in blocks of eight so
`K=32` does not require one monolithic decoder graph. All block gradients
accumulate before one ordinary Adam step; there is no stationary-gradient,
monotonicity, rollback or restart condition. A focused regression test checks energy
and coordinate-gradient equality against the monolithic objective. `K=64`
remains a runtime sensitivity only. V2 used eager decoder-energy evaluation.

#### Calibration v2 outcome

Private version `maximshtefan/eqvae-fg-stage-a2-calibration/2` completed in
`35906.99` seconds with both workers successful. Its common decision remains
`unresolved` (selection SHA-256
`4ae2b22f98419fe2ddb3777b14a53a49b834a3e7186c8a776d86c9523e40a4b1`).
All runtime probes through `K=64` completed within the fixed memory ceiling;
dimensions `16/32`, epsilon `.016` and Adam rate `.005` satisfy their partial
criteria.

Two preregistered criteria fail. No radius meets the worst-case `.05`
linearization bound: `.0015` is best at `.061422`, although 1190/1200 samples
pass and p99 is `.048893`; the ten exceedances are prescribed-`SO(2)` midpoint
directions. Every one of the 32 optimizer candidates reduces energy by at least
`5.046%` without trust contact, but there is no common recorded milestone
within 1% of all candidates' individual best energies, and six candidates first
reach their best at the 384-iteration ceiling. Consequently the emitted blocker
name `optimizer_did_not_improve_within_grid` must be read as failure of the
common-near-best milestone rule, not absence of optimization progress. These
calibration outputs are not Stage A2 scientific evidence and do not unlock the
final-pilot inputs.

#### Calibration v3 contract

V3 resolves only the remaining solver search-space and finite-budget questions;
it does not repeat the complete v2 diagnostics. The same frozen models receive
the same candidates. The two fixed workloads are rank 4 prescribed and rank 16
encoded, which cover both endpoint routes and were the hardest complementary
128-step cases in v2. At `K=16`, compare nested midpoint-visible affine charts
`d=32,128` with a full-latent control. At the harder rank 4 prescribed workload,
repeat `d=128` and full latent at `K=32`. Every candidate creates Adam once and
runs continuously for 128 steps at learning rate `.005`, with milestones
`0,16,32,64,96,128`. `.005` is the `d=32` reference rate; each candidate uses
`.005 sqrt(32/d)` so Adam's expected coordinate-step norm does not grow merely
because the parameterization contains more coordinates (`d=16384` for full
latent).

The full-latent control directly optimizes the `K-1` normalized latent offsets
of shape `16x32x32`; it never materializes a `16384x16384` identity. The reduced
charts share one nested `d=128` basis per endpoint pair. V3 selects `d=128` only
if its thin metric remains conditioned on the initialization line, every common
candidate improves by at least 1%, finishes within 1% of its best observed
milestone energy, avoids the trust boundary, and its best recorded energy is
within 1% of the equal-budget full-latent control at both discretizations.
Otherwise it selects
the full-latent decoder-energy parameterization only if that control itself
meets the common finite-budget criteria. `d=32` remains a recorded sensitivity.

Only the repeated eight-edge scalar closure is eligible for compilation:
`latents -> decoder -> decoder-edge energy`. JVP/VJP chart construction remains
eager. Because the models are frozen, the `SO(2)` decoder materializes its exact
dense equivariant kernels once after checkpoint load. On one real block, cached
and uncached outputs, energies and latent gradients must agree exactly; compiled
energy and latent-gradient relative errors must be at most `1e-5` and `1e-4`.
The static full graph is used only when its settled T4 time beats eager. The
compiler choice changes execution, not the objective, optimizer, coordinates or
budget. The complete run has a 120-minute wall-time ceiling.

V2's failed sampled first-order radius rule is retained as its historical
outcome, not silently reinterpreted. It is not repeated in v3: an affine chart
is an exact smooth map, while decoder nonlinearity is part of its pullback
metric rather than an error against a linear decoder approximation. Scientific
Riemannian labels still require the later pathwise rank, conditioning, trust,
and knot/chart-refinement checks defined below.

### Objective

Determine independently whether:

1. `e_k` and `a_k` are the same latent state;
2. they are different representatives of one decoder-equivalence class;
3. independently solved decoded paths are covariant under exact `C4`;
4. the four cardinal anchors lie on one constant-speed closed geodesic with
   exact-`C4` covariance;
5. one quarter-path and its transported tangent can generate the remaining
   three quarters.

These are separate claims. None is allowed to rescue another.

### Information regimes

Keep three analyses separate:

1. **Endpoint-known local interpolation.** Each side may use its two literal
   endpoints and decoder differentials along the candidate path. The other two
   anchors and any desired intermediate rotation image remain unavailable to
   that side's objective and initialization.
2. **Free local continuation.** Only `z_0`, `z_1`, their solved first-side Log
   direction, and decoder differentials along the rollout are available.
   `z_2`, `z_3`, their decodes, all statistics derived from them, and every
   same-WSI fitted object remain sealed until `hat z_2,hat z_3,hat z_4` are
   written.
3. **Cross-patch transfer.** Excluded from Stage A2. A later separately locked
   analysis may fit a shared object on the pilot-safe WSI pool, but cannot
   define, initialize, tune, select, or rescue either local analysis.

Using all four anchors to concatenate four endpoint-known sides is a
teacher-forced description. Using all four anchors to select a chart and then
claiming that the chart predicted the cycle is prohibited.

### Fixed local charts and representative separation

A single pooled linear `U` is not the primary manifold model. Every candidate
Riemannian solve uses one fixed `C2` chart `phi` with an orthonormal Jacobian
basis, a declared coordinate domain, and a preregistered trust radius. The
first implementation may use an affine `C-infinity` chart
`phi(xi)=z_ref+U xi`; it must not call independently changing knot frames an
atlas. A deterministic calibration may construct nested candidate `U(d)` from
the permitted endpoints, decoder JVP/VJP, and decoder-energy gradients on the
latent-line initialization. Each candidate is then frozen before optimization.
Increasing `d` creates a separate sensitivity solve rather than silently
changing coordinates inside one solve.

The trust region limits coordinate radius, distance from the permitted local
initialization, and per-step motion using model-independent dimensionless
scales derived only from permitted endpoint secants. A path that touches the
trust boundary without convergence loses the word `local`; the unrestricted
solution is retained only as a global decoder-energy sensitivity with latent
norm, raw-output range, sharpness, content drift, and rank diagnostics.

Two endpoint contracts are required:

1. **Representative-conditioned path:** both `z_L` and `z_R` are literal.
   Vertical or `epsilon`-inactive motion is allowed and measured. This is
   always called a decoder-energy path, not a quotient geodesic.
2. **Class-target path:** fix `gamma(0)=z_L`, require
   `D(gamma(1))~=D(z_R)` under the endpoint criterion, and allow the terminal latent
   `z_R_prime` to differ from `z_R`. Only if the fixed-chart and full-rank criteria
   pass may this be called a reduced Riemannian geodesic. The residual
   `z_R_prime-z_R` is not called gauge until a separate connected-bridge test
   validates approximate decoder equivalence.

At a valid chart knot, let `Q_j` be an orthonormal frame for the complete fixed
chart tangent and `C_j=J_D(z_j)Q_j` its decoder-image tangent frame. Every
chart direction participates in `G_phi`, the geodesic, and transport. If any
chart singular value falls below the preregistered conditioning threshold, reject
that chart; do not drop the direction inside the solve or insert an eigenvalue
floor. Exact null and positive-but-small `epsilon`-inactive directions are
recorded separately as diagnostics. Any rank or conditioning failure
invalidates Exp/Log, Levi-Civita, and holonomy labels for that path.

Fiber searches use a distinct joint search space containing both literal endpoints,
their secant, target-free exact-null candidates when identifiable, declared
`epsilon`-inactive candidates, and visible corrections. A Riemannian chart that
excluded inactive directions must not be reused as if it could discover them.
Conversely, an inactive bridge direction cannot lower a reduced geodesic merely
by changing its representative.

### A. Exact anchors and decoder-fiber bridges

Persist `e_k`, `a_k`, `D(e_k)`, and `D(a_k)` for ranks 0 and 12. Reproduce the
direct exact-`C4` action residuals and add the missing comparison `D(e_k)`
versus `D(a_k)`. The equality `e_0=a_0` is an exact no-op sanity row; do not
optimize a degenerate `k=0` bridge.

For each nonidentity `k`, first require the normalized endpoint decoded RMS to
pass the bridge-assessability criterion. Then optimize a joint-chart bridge
`beta_k:e_k -> a_k` under

`E_D(beta)=sum_j ||D(beta_(j+1))-D(beta_j)||_X^2/delta_t`.

Use exactly the preregistered latent line and two seeded, secant-normalized
Rademacher perturbation starts. Persist every result. The canonical reported
candidate is the lowest-energy converged path that passes endpoint, decoded-
tube, reversal, and knot/chart-refinement criteria; no target retained elsewhere
may break a tie. Report endpoint decoded RMS, bridge energy and length, maximum
decoded diameter relative to both endpoint decodes, latent length, exact-null
and `epsilon`-inactive usage, active rank, reversal, refinement, and distinct
solutions found under those starts.

A large latent displacement with a connected negligible-output-motion bridge
supports approximate decoder equivalence under the operational tolerance; it
does not prove an exact fiber. Similar endpoint decodes without a connected
bridge do not. Failure to find a bridge is `unresolved/not found`, never
evidence that no bridge exists. The equal-budget full-latent control detects
reduced-chart search bias; only a separately established mathematical lower
bound or global certificate could exclude bridge existence.

### B. Four independently optimized sides

For both the encoded and prescribed cycles, independently optimize

`gamma_k:z_k -> z_((k+1) mod 4)`, `k=0,1,2,3`.

Solve both the literal representative-conditioned and free-terminal
class-target contracts. The primary objective is unregularized decoder energy
inside the preregistered local trust region; the same objective without the
trust region is a labelled global sensitivity, not a fallback selected from
its result. Use the same three preregistered starts as the bridge protocol.
Use `K=16` and mandatory `K=32` refinement; `K=64` is a fixed first-side knot
sensitivity. Run an equal-budget full-latent control for every side and
endpoint contract used by a cycle, covariance, closure, or holonomy decision.
If that complete control is computationally excluded by the locked budget,
the affected reduced-chart conclusion remains unresolved rather than being
inferred from the first side.

No intermediate desired rotation image, other side, cross-patch basis, or
withheld anchor enters the objective, initialization, chart/dimension choice,
stopping rule, or branch selection. Persist all solutions found under the
fixed starts and cluster distinct paths with a preregistered decoded-curve
distance; do not claim exhaustive Logs or global uniqueness.

Report every decoded knot, endpoint error, energy, length, constant-speed
error, reversal, knot/chart/full-latent refinement, decoded sharpness and
content drift, latent norm, exact-null and `epsilon`-inactive velocity,
rank/conditioning, and agreement across starts. Raw length and energy are
reported beside `L_D/||D(z_R)-D(z_L)||_X` and
`E_D/||D(z_R)-D(z_L)||_X^2`, with a preregistered endpoint-signal degeneracy
criterion. Persist contact sheets and animations or ordered frames so the
intermediate interpolation is directly inspectable.

### C. Exact-C4 covariance of complete paths

Test every side rather than generating three sides from the first and assuming
the result:

`D(gamma_(k+1)(t)) ~= R_90 D(gamma_k(t))`.

This uses only exact array permutations. For the prescribed cycle,
`rho(r)a_k=a_(k+1)` is true by construction; empirical evidence concerns the
decoder paths, not discovery of that latent group law. Where the differential
action-isometry criterion passes, also compare

`[gamma_(k+1)(t)]_D` with `[rho(r)gamma_k(t)]_D`.

At every valid knot, let `Q_z` and `Q_rz` be the complete source and target
chart-tangent frames, and define

`T_r=J_D(rho(r)z) d rho(r) Q_z`, `S_r=R_90 J_D(z)Q_z`.

The model-independent criterion includes all of:

- worst-direction commutation
  `||T_r-S_r||_2/(||S_r||_2+epsilon)`;
- metric distortion
  `||T_r^T T_r-S_r^T S_r||_2/(||S_r^T S_r||_2+epsilon)`;
- chart-subspace covariance
  `||(I-Q_rz Q_rz^T)d rho(r)Q_z||_2` and every principal angle between
  `d rho(r)Q_z` and `span(Q_rz)`;
- pointwise decoder commutation, equal chart dimension/rank, and preservation
  under knot and chart refinement.

The machine contract fixes every threshold. An averaged Frobenius residual or
exact endpoint commutation alone cannot establish differential isometry.

Measure endpoint and tangent continuity, equality of side length/energy,
decoded closure, and fourfold composition. If independently solved sides have
tangent jumps, the concatenation is only a closed piecewise path, not one
smooth closed geodesic.

### D. Free geodesic continuation

For each encoded and prescribed cycle whose class-target first side passes all
restricted-immersion Riemannian criteria, solve

`v_0=Log^phi_(z_0)(hat z_1)`, with `D(hat z_1)~=D(z_1)`,

using only its permitted information. The machine contract must fix the IVP
geodesic integrator, order, step schedule, chart evaluation, velocity update,
retraction, rank-change behavior, and time refinement. Continue the same
geodesic and velocity for three further equal quarter durations to produce

`hat z_2=Exp^phi_(z_0)(2 v_0)`, `hat z_3=Exp^phi_(z_0)(3 v_0)`, and
`hat z_4=Exp^phi_(z_0)(4 v_0)`.

This notation denotes numerical continuation of one trajectory, not four
independent exponentials in a fixed linear chart. Compute the first side from
`z_0,z_1`, then continue to `hat z_2,hat z_3,hat z_4` without using `z_2,z_3`,
the future action, a closure loss, target-aware stopping, or target-aware branch
selection. Only after the predictions exist, compare them with the withheld
anchors and evaluate closure. This sequencing may live in one simple runner;
it does not require separate workers, receipts, or a journaling layer.

Also retain `hat z_1=gamma(1)`, the class-target terminal
representative actually reached by the trajectory. It is not silently replaced
by literal `z_1` in any continuation or downstream diagnostic.

The continuation state may retain the fixed chart origin and numerical scales
required by the IVP, but no functional of return distance to `z_0` may affect
steps, stopping, enrichment, branch clustering, or selection after the first
quarter. Without an accepted post-hoc bridge, decoded agreement is reported
only as decoded agreement, not quotient closure or gauge motion. A
teacher-forced diagnostic may restart from each observed anchor but is never
pooled with or called free continuation.

Continue every distinct first-side solution found under the preregistered
starts and refinements. This does not establish that all mathematical Logs were
found or that the Log is unique. Chart failure, rank change, or multiple
scientifically different continuations is reported as unresolved rather than
selecting the branch closest to a withheld or closure anchor.

### E. Parallel transport and closed-loop holonomy

Moving a point along `gamma_k` is geodesic motion. Parallel transport moves a
tangent vector. Let

`v_k=Log^phi_k_(z_k)(hat z_(k+1)^k)`, where
`D(hat z_(k+1)^k)~=D(z_(k+1))`.

Compare separately

`P_(gamma_k)v_k`, `d rho(r)v_k`, and `v_(k+1)`.

The transported tangent ends at the class-target representative
`hat z_(k+1)^k`, whereas the next independently solved tangent begins at the
literal `z_(k+1)`. Therefore every intrinsic comparison with `v_(k+1)` first
requires an accepted bridge plus preregistered tangent-space identification
between those base points. For the prescribed cycle, `d rho(r)v_k` is based at
literal `z_(k+1)` by construction and needs the same identification before it
can be compared with the transported endpoint. For the encoded cycle,
`rho(r)e_k` is generally not even literal `e_(k+1)`, so an additional accepted
identification is required. Without all required criteria, report decoded vectors
in the common ambient output space only as an extrinsic return defect and make
no intrinsic angle, continuity, or transport-agreement claim.

At adjacent path knots of one fixed valid chart, use the first-order
induced-metric projection

`c_(j+1)=C_(j+1)^dagger C_j c_j`,

where the latent tangent is `u_j=Q_j c_j`, its decoded tangent is `C_j c_j`,
and `u_(j+1)=Q_(j+1)c_(j+1)`. Coordinates, latent tangents, and decoded
tangents are persisted separately.

`C_j` has full column rank by the chart criterion, and the pseudoinverse uses the
complete chart tangent. Do not truncate positive modes or hide failed
directions with an eigenvalue floor. Check metric-norm preservation,
forward/backward error, knot and frame refinement, tangent angles, and action
covariance. Preservation of a geodesic's own velocity under the same connection
is a numerical implementation check, not independent scientific evidence. The
independent evidence is agreement with sealed future anchors, separately solved
side Logs, or `d rho(r)` where their base points are validly identified.

The teacher-forced holonomy diagnostic uses one separate, fixed, cycle-local
`C2` chart containing all four literal anchors and re-solves all four sides
inside that one chart. It never transports across independently chosen
side-chart coordinates. Because it uses all anchors, it is descriptive only
and cannot enter free continuation. Holonomy is defined only if this chart has
constant full column rank along an exactly closed loop and the side junctions
admit the declared tangent-continuity classification.
The free rollout instead reports point, tangent, and frame **return defects**.
It is not assigned a holonomy operator merely because its endpoint decode is
close to `D(z_0)`. A later explicitly closed comparison may be reported only if
an accepted connected bridge and a preregistered compatible tangent-space
identification make the complete return path well defined.

Report:

- **point closure/return defect:** raw latent, decoded-output, and—only after an
  accepted connected bridge—operational decoder-equivalence discrepancy;
- **rotation-tangent return:** metric norm, cosine/angle, and decoded-tangent
  discrepancy between the transported tangent and the initial tangent;
- **frame return map:** start from an output-orthonormal complete chart frame, transport
  it, and record the raw return matrix `M`, out-of-start-subspace leakage,
  singular values, and `||M^T M-I||` before any orthogonalization;
- **orthogonal holonomy estimator:** only for a valid closed loop, report the
  preregistered polar factor of nonsingular `M`, its determinant, principal
  rotation angles, and knot/chart refinement error. No free Procrustes
  alignment may rotate away the measured holonomy.

Identity rotation-tangent holonomy is required for a smooth closed geodesic
generator. Full-frame identity is not required: nontrivial full-frame
holonomy can reflect curvature. If side junctions are not tangent-continuous,
report piecewise-loop transport/return without relabelling it as smooth-
geodesic holonomy.

### F. Secondary exact-C4 sector diagnostic

After primary local outputs are frozen, compute the four-point `q` decomposition
of the observed tuple `[z_0,z_1,z_2,z_3]` and the frozen free tuple
`[z_0,hat z_1,hat z_2,hat z_3]`. Normalize each sector's squared norm by total
centered four-tuple energy, apply a degeneracy criterion, and report sector-wise
prediction error. Exact reconstruction of an observed four-tuple by its DFT is
an identity check, not a result.

No pooled sector basis or cross-patch fitted action belongs to Stage A2. Such a
test requires its own transfer estimator, block quotas, equal achieved
dimension, baselines, and budget fixed before outputs. If run later, it
uses only the pilot-safe ten-WSI pool and cannot alter or rescue local results.

## Interpretation

| Observation | Local interpretation |
| --- | --- |
| `e_k-a_k` is large but connected bridge criteria pass | Different encoder/action representatives are approximately decoder-equivalent under the operational tolerance. |
| Complete sides are exact-`C4`-covariant but free continuation misses anchors | The independently solved paths share cardinal covariance, but the first geodesic branch does not generate the later anchors. |
| Free continuation predicts `z_2,z_3,z_0`, closes, and agrees with independent side Logs | Strong evidence for a local closed geodesic, covariant under `C4`, through the rotated anchors. It is not yet a continuous group orbit. |
| Prescribed cycle is clean and encoded cycle has accepted representative bridges | Decoder rotation geometry is coherent while the encoder may choose a different local section. |
| Free continuation fades while the prescribed action rotates | The prescribed action does not coincide with continuation of the geodesic branch found from the first quarter. |
| Fiber bridges visibly change decoded content | Encoder/action disagreement is not explained only by decoder redundancy. |
| Restricted-chart rank changes along a path | Only decoder-energy evidence is defined there. |
| Point closes under an accepted bridge and the rotation tangent returns under a valid identification | Local operational decoder-equivalence closure and a persistent restricted-geodesic generator are plausible. |
| Point closes but the chart-tangent rotation direction does not | The loop is closed but not a smooth geodesic cycle with one persistent generator. |
| Rotation tangent returns but the remaining frame has holonomy | The cycle direction is coherent while surrounding chart directions record curvature of the restricted immersion. |
| Predicted and observed `q` sectors agree | The free four-state tuple reproduces that exact-`C4` Fourier summary; the DFT identity itself supplies no evidence. |

## Implementation And Outputs

- The primary output map is `Phi(z)=D(z)` in raw unclamped FP32 RGB. Its inner
  product is the mean over all `3x256x256` scalars, so norms are per-scalar RMS.
  Clamp never enters `G`, energy, length, bridge, transport, or closure.
  Centered-disk/crop and clamped-image measurements are labelled boundary and
  visual-artifact sensitivities, not alternative selectable metrics.
- Run one frozen model per T4 in separate spawned processes; the parent does not
  initialize CUDA.
- Reverse AD may optimize latent/chart knots, but model parameters never
  accumulate gradients and graphs are released after each work unit.
- Batch compatible knots or sides only when optimizer and scientific paths
  remain independent.
- Print enough progress to identify the current model, patch, route, iteration,
  elapsed time, and CUDA peak memory.
- Produce one result JSON plus the decoded intermediate images and compact
  plots needed to inspect paths, return, and holonomy.

Before Stage A2, its machine contract fixes only the numerical values actually
used: chart construction and dimension, trust radius, rank/conditioning
thresholds, optimizer and iterations, path discretization, IVP integrator,
starts, refinements, and comparison tolerances. Both models receive the same
settings and compute budget.

Focused tests are limited to numerical seams where a plausible bug changes the
result: endpoint handling, energy/gradient equivalence, target isolation, known
synthetic Exp/Log and holonomy cases, exact-`C4` covariance, and transport
coordinate consistency. Do not mirror the complete contract in tests and do
not run unrelated repository tests.

## Later Work

- Expand geodesic/action analysis to the five evaluation WSIs only after the
  pilot fixes a numerically valid local chart family, equivalence rule, and
  solver.
- A continuous-angle experiment must first quantify padding and interpolation
  error. Exact `C4` alone cannot select a unique continuous generator.
- A shared learned action must fit only on fit WSIs and transfer to non-cardinal
  evaluation views without seeing their target encodings.
- Posterior Wasserstein/Fisher geometry remains separate from mean-field
  decoder geometry.
- Cross-content interpolation is descriptive until content coordinates and
  transfer criteria are fixed.
- PGA requires a validated chart plus stable Exp/Log maps. Exact-`C4` quotient
  PGA additionally requires the action to be an isometry.
- Connection, holonomy, gauge synchronization, vector diffusion, and transport
  support only local bundle compatibility; stabilizers and ambiguous phases
  remain explicit. Holonomy is not torsion.
- After Stage A2, a separate sampled-manifold experiment may compare
  connection-Laplacian or GEOMANCER structure in raw image space, encoder
  space, and the decoder-visible restricted geometry. It requires dense local
  neighborhoods over angle and content; four cardinal points are insufficient.
  Because the one-dimensional `SO(2)~=S1` factor has trivial Levi-Civita
  holonomy, failure of unsupervised GEOMANCER factorization cannot refute a
  pose orbit. Validate the method first on synthetic products with known
  nontrivial holonomy.
- A `D4` claim requires every reflection and group relation, not one favorable
  transform.

## References

- Arvanitidis, Hansen, and Hauberg, *Latent Space Oddity*.
- Shao, Kumar, and Fletcher, *The Riemannian Geometry of Deep Generative
  Models*.
- Singer and Wu, *Vector Diffusion Maps and the Connection Laplacian*.
- Pfau et al., *Disentangling by Subspace Diffusion* (GEOMANCER).
- Fletcher et al., *Principal Geodesic Analysis for the Study of Nonlinear
  Statistics of Shape*.
