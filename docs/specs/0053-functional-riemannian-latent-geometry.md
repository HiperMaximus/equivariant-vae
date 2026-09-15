# Spec 0053: Functional And Riemannian Latent Geometry

Status: active; numerical method and Stage A1 accepted, exact-`C4`
geodesic/fiber experiment next
Owner/workstream: frozen normal versus continuous-`SO(2)` VAE latent analysis
Last updated: 2026-09-15

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
| Weight dataset | `maximshtefan/eqvae-vae-test-reconstruction-inputs-v1` |
| Patch dataset | `maximusshtefan/patches-pre-shuffled-ubc-ocean` |
| Fixed patches | The accepted ordered validation 25; selector SHA-256 `ace244ecdd67aaa1ebc7d08065f1e4bfa3c0d54806d4f3b50fb38a3ae000447f` |
| Input | FP32 `3x256x256` normalized image |
| Latent state | Deterministic posterior mean `mu`, FP32 `16x32x32`; `logvar` is telemetry unless posterior geometry is explicitly studied |
| Exact group | `C4={e,r,r^2,r^3}`, `r=torch.rot90(+1)` |
| Primary pilot patches | Fixed ranks 0 and 12 |

Both checkpoints remain frozen. No training, sealed-test access, new example
selection, or model-dependent retuning is allowed.

The existing WSI-disjoint post-hoc split is retained for later fitted work:

- evaluation WSIs `5970,60988,38349,11417,10077`, ranks
  `3,2,0,1,11,6,7,9`;
- fit WSIs `39880,46444,23629,22654,59031,27747,26025,31297,21260,39252,57265`,
  ranks `4,5,8,10,12,14,13,16,18,19,20,21,22,15,17,23,24`.

“Fit” means fitting post-hoc bases or actions, never VAE training. The split is
WSI-disjoint but not an untouched validation set because earlier fixed-25
outputs are already known.

## Scientific Boundaries

- Four cardinal samples identify only exact `C4`, not a unique continuous
  generator or winding number.
- Four centered points span at most three dimensions by construction; their
  per-patch PCA rank is not evidence for a low-dimensional rotation manifold.
- A closed curve is not automatically smooth, an action orbit is not
  automatically geodesic, and a minimum-energy path is not automatically
  semantic.
- Decoder pullback geometry can be degenerate. Riemannian language is used only
  on a stable positive-definite horizontal chart.
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

Its local vertical space is `V_z=ker J_D(z)`. Two distant latent states can be
gauge-equivalent when a connected path between them produces negligible
decoded motion.

A pose orbit `{rho(g)z}` is different: it normally moves across decoder
fibers over differently rotated outputs. Only in a separately established
content-versus-pose quotient may a rotation orbit be interpreted as a pose
fiber.

Operationally, corresponding encoded and prescribed representatives form

```text
e_k = E(R^k x)  --- decoder-null/gauge bridge ---  a_k = rho^k E(x)
       | D                                               | D
       v                                                 v
    D(e_k)                ~= R^k D(e_0)               D(a_k)
```

Similar endpoint decodes are insufficient to establish one connected fiber.
The experiment also requires a low-decoder-energy bridge between the
representatives.

### Decoder pullback and horizontal quotient

The decoder pullback form is

`g_z(u,v)=<J_D(z)u,J_D(z)v>`,

with `G(z)=J_D(z)^T J_D(z)`. It is positive semidefinite in the full latent
space because vertical decoder-null directions have zero length.

For a reduced chart basis `U`,

`B(z)=J_D(z)U`,

`G_U(z)=B(z)^T B(z)`.

Rank and conditioning are determined from the thin operator `B`, not from
squaring its condition number and guessing from noisy eigenvalues of `G_U`.
When `B` has stable rank along a path, its active directions define the local
horizontal quotient. A path there may be called a reduced Riemannian geodesic.
If rank is unstable, the same computation is reported only as a
decoder-energy path.

Latent paths may differ while their decoded curves agree. Such multiplicity is
possible gauge evidence and remains visible in the results.

## Accepted Numerical Method

Decoder differential calculations use direct disposable JVP/VJP calls with
model parameters frozen:

`Gv=J_D(z)^T(J_D(z)v)`.

The accepted settings are:

- eager FP32 operator arithmetic with `torch.compile` disabled;
- finite-difference JVP reference epsilon `0.008`;
- direct graphs released after every work unit;
- microbatch 4 for compatible independent directions;
- matrix-free SLQ with Lanczos depth `m=64`;
- `r=64` independent normalized Rademacher probes;
- FP64 tridiagonal eigensolves and 99% scalar intervals.

`m` controls Krylov depth per SLQ probe and `r` controls independent
replication. Neither is a latent projection dimension or a geodesic batch
size. A full Jacobian or full `G` is never materialized. Small reduced
`G_U` matrices are computed exactly when geodesic or transport calculations
need them.

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

Final reproducible implementation:
`kaggle/kernels/functional_geometry_stage_a1_c4_slq/`.
Accepted output:
`runs/kaggle/functional_geometry_stage_a1_c4_slq_v1/functional_geometry_stage_a1_c4_slq_v1/stage_a1_result.json`,
SHA-256
`f59bf4eb7bba136c02cca24b237414da1dca23e4b94b781c4eb3d5978840e449`.

## Current Experiment: Exact-C4 Geodesics, Transport, And Fibers

### Objective

Determine independently whether:

1. `e_k` and `a_k` are the same latent state;
2. they are different representatives of one decoder-equivalence class;
3. successive cardinal states form a coherent exact-`C4` action;
4. that action orbit is also a constant-speed closed geodesic;
5. one quarter-path and its transported tangent can generate the remaining
   three quarters.

These are separate claims. None is allowed to rescue another.

### Reduced charts with literal anchors

Fit a nested shared exact-`C4` sector basis using only the 11 fit WSIs.
Primary shared dimension is `d=32`; `d=16,64` are fixed sensitivities.

The construction is deterministic and separate for each VAE. For every fit
four-tuple `z_0,...,z_3`, form the orthonormal real four-point DFT coefficients

`c_0=(z_0+z_1+z_2+z_3)/2`,

`c_2=(z_0-z_1+z_2-z_3)/2`,

`c_c=(z_0-z_2)/sqrt(2)`, and `c_s=(z_1-z_3)/sqrt(2)`.

These are respectively the real `q=0`, `q=2`, and paired `q=+/-1` sectors.
Stack coefficients from both encoded and prescribed cycles using fit WSIs
only, center each sector with fit statistics, and take deterministic thin-SVD
directions. Merge candidates by decreasing fit singular energy with ties
ordered `q=0,q=2,q=+/-1`; whenever a `q=+/-1` direction is selected, adjoin
and orthonormalize its exact `rho(r)` partner so the pair is never split. One
ordered candidate list supplies the nested prefixes `d=16,32,64`; evaluation
anchors never affect its means, ordering, signs, pair orientation or rank.

For each anchored cycle, augment the frozen shared basis with the at most three
independent offsets from its first anchor. This puts all four `e_k`, or all
four `a_k`, literally in the chart instead of projecting the endpoints.

The free continuation uses only the frozen fit basis and `z_1-z_0`. It must
not use `z_2`, `z_3`, their decodes, or a basis fitted from them.

### A. Exact anchors and decoder-fiber bridges

Persist all `e_k`, `a_k`, `D(e_k)`, and `D(a_k)`. Reproduce the direct
exact-`C4` action residuals and add the missing comparison
`D(e_k)` versus `D(a_k)`.

For every `k`, optimize an unregularized bridge
`beta_k:e_k -> a_k` under

`E_D(beta)=sum_j ||D(beta_(j+1))-D(beta_j)||_X^2/delta_t`.

Report endpoint decoded RMS, bridge energy and length, maximum decoded
diameter, latent length, active rank, near-null usage, reversal, refinement,
and multiplicity across target-free starts. A large latent bridge with
negligible connected decoded motion supports approximate decoder-fiber
equivalence.

### B. Four independently optimized sides

For both the encoded and prescribed cycles, independently optimize

`gamma_k:z_k -> z_((k+1) mod 4)`, `k=0,1,2,3`.

Use unpenalized decoder energy first, with the latent line and two seeded
target-free perturbations as fixed starts. Use `K=16` and mandatory `K=32`
refinement. `K=64` is a fixed first-side sensitivity for ranks 0 and 12.
No intermediate desired rotation image enters the objective or initialization.

Report endpoint error, energy, length, constant-speed error, reversal,
refinement, decoded sharpness and content drift, latent norm,
rank/conditioning, and agreement across starts.

### C. Exact-C4 covariance of complete paths

Test every side rather than generating three sides from the first and assuming
the result:

`D(gamma_(k+1)(t)) ~= R_90 D(gamma_k(t))`.

This uses only exact array permutations. Where the action-isometry gate passes,
also compare

`[gamma_(k+1)(t)]_D` with `[rho(r)gamma_k(t)]_D`.

Measure endpoint and tangent continuity, equality of side length/energy,
decoded closure, and fourfold composition.

### D. Parallel transport of the rotation tangent

Moving a point along `gamma_k` is geodesic motion. Parallel transport moves a
tangent vector. Let

`v_k=Log_[z_k]([z_(k+1)])`.

Compare separately

`P_(gamma_k)v_k`, `d rho(r)v_k`, and `v_(k+1)`.

Agreement is stronger than action coherence: it asks whether the rotation
orbit itself is geodesic and its generator persists from quarter to quarter.

At adjacent path knots, use the first-order induced-metric projection

`v_(j+1)=G_U(z_(j+1))^dagger B(z_(j+1))^T B(z_j)v_j`.

The pseudoinverse acts only on the declared active horizontal spectrum; do not
hide vertical modes with an eigenvalue floor. Check metric-norm preservation,
forward/backward error, knot refinement, tangent angles, and the full-loop
transported tangent. Full-frame holonomy is measured, not required to vanish.

### E. Free geodesic continuation

Use only `z_0`, `z_1`, their fitted log direction, and fit-WSI chart state.
Transport the terminal velocity and apply the reduced exponential/geodesic
integrator for the same arc length to predict

`hat z_2`, `hat z_3`, and `hat z_4`.

Only after prediction compare them with the unused exact anchors
`z_2,z_3,z_0`, both in latent coordinates and modulo decoder fibers. Report
quarter-step decoder-quotient error, accumulated speed and norm drift, decoded
anchor RMS, and latent/decoded closure of `hat z_4`.

Run the free continuation separately for the encoded and prescribed cycles. A
teacher-forced diagnostic may restart from each observed anchor but is never
pooled with the free result.

## Interpretation

| Observation | Local interpretation |
| --- | --- |
| `e_k-a_k` is large but the bridges are decoder-null | Encoder representatives differ by gauge; quotient equivariance is plausible. |
| Complete sides are exact-`C4` copies but free continuation misses anchors | A coherent action may exist, but its orbit is not decoder-geodesic. |
| Free continuation predicts `z_2,z_3,z_0` with stable transport | Strong evidence for a local closed geodesic rotation orbit in the decoder quotient. |
| Prescribed cycle is clean and encoded cycle gauge-irregular | Decoder rotation geometry is coherent while the encoder chooses a non-equivariant section. |
| Free geodesic fades while the prescribed action rotates | Rotation exists but is not the decoder-metric shortest path. |
| Fiber bridges visibly change decoded content | Encoder/action disagreement is not explained only by decoder redundancy. |
| Horizontal rank changes along a path | Only decoder-energy evidence is defined there. |

## Implementation, Observability, And Outputs

- Run one frozen model per T4 in separate spawned processes; the parent does not
  initialize CUDA.
- Reverse AD may optimize latent/chart knots, but model parameters never
  accumulate gradients and graphs are released after each work unit.
- Batch compatible knots or sides only when optimizer and scientific paths
  remain independent.
- Log run, worker, model, patch, cycle, bridge, edge, start, refinement,
  optimizer checkpoint, transport quarter, output write, elapsed time, and CUDA
  allocated/reserved/peak memory.
- Every exception records its type, message, active and last-completed work
  units, and full traceback. Partial completed units remain readable.
- Produce one result JSON, one JSONL event log, exact-anchor tables, decoded
  knot contact sheets, and path-difference panels for every model/rank/cycle.

Before remote execution, a compact machine contract must fix the chart
construction, active-rank rule, optimizer, call/iteration ceiling, seeds, and
numerical tolerances without using named-model outcomes. Focused tests cover
literal endpoints, target isolation in free rollout, vertical-null removal,
Euclidean transport, exact-`C4` covariance, reversal, and refinement. Do not
run unrelated repository tests.

## Later Gated Work

- Expand geodesic/action analysis to the five evaluation WSIs only after the
  pilot fixes a numerically valid chart and solver.
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
- Connection, gauge synchronization, vector diffusion, and transport support
  only local bundle compatibility; stabilizers and ambiguous phases remain
  explicit.
- A `D4` claim requires every reflection and group relation, not one favorable
  transform.

## References

- Arvanitidis, Hansen, and Hauberg, *Latent Space Oddity*.
- Shao, Kumar, and Fletcher, *The Riemannian Geometry of Deep Generative
  Models*.
- Singer and Wu, *Vector Diffusion Maps and the Connection Laplacian*.
- Fletcher et al., *Principal Geodesic Analysis for the Study of Nonlinear
  Statistics of Shape*.
