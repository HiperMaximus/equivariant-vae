# Spec 0012: Continuous SO(2) VAE Architecture

Status: radial/layout oracle complete / equal-copy F01 fixed / Spec 0013 accepted
Full-VAE readiness: mechanics ready / coding requires separate authorization
Owner/workstream: matched continuous-`SO(2)` VAE
Last updated: 2026-08-13

## Purpose

Define the first repo-owned, compile-oriented continuous-`SO(2)` VAE before
code is written. It must preserve the normal-VAE experiment's macro topology
while using no more than its 3,958,435 learned parameters. It is a single
selected architecture for one known hardware/runtime setup, not a reusable
equivariance library. It authorizes only the small basis-oracle slice described
below, not the full VAE or training.

## Measured Basis-Oracle Result

The one-off oracle is complete. Its source, selected model input, and numerical
evidence are:

```text
src/eqvae/models/so2_basis.py
scripts/select_so2_basis.py
configs/spec0012/so2_basis_manifest.json
docs/data/spec0012_so2_basis_audit.json
tests/test_so2_basis.py
```

The selected basis outcome, equal-copy F0/F1 layout, support assignment, and
compiled convolution mechanics are locked and accepted. The
oracle's global profiles were:

| profile | centres | widths | qmax | result |
| --- | --- | --- | --- | --- |
| `7-low` | `[1,1.90395977,2.75]` | `[.3,.3,.3]` | `[2,2,2]` | pass |
| `7-full` | none | none | none | locked grid has no legal seed |
| `9-low` | `[1,1.99907757,2.87711643,3.75]` | `[.3,.3,.3,.3]` | `[2,2,2,4]` | pass |
| `9-full` | `[2,2.64125343,3.25768854,3.75]` | `[.3,.3,.3,.3]` | `[4,4,4,4]` | raw basis pass; F2 sampling gate fail |

The three feasible profiles pass every raw rank/conditioning gate and all nine
one-copy `escnn` span/kernel/equivariance comparisons. `9-full` is also robust
over all 575 retained simultaneous perturbations and adds `D_high=24`
incremental q3/q4 dimensions. Those incremental subspaces match the pinned
`escnn` reference with worst projector distance `8.01e-8`. Nevertheless its
sampled-grid result is `E_high=2.0000585868`, while
`E_floor=0.1135105892` and `E_limit=0.1702658838`; the locked decision rule
therefore rejects F2 and selects F01. The audit retains every pair/angle
subspace error and identifies `F2->F2` at 45 degrees as the worst case; exact
90-degree errors are at most `2.74e-14`.

The locked `7-full` premise is partially wrong. Two shells must retain q4 and
therefore have `r>=2`; within the 7x7 upper bound `2.75`, the declared coarse
grid supplies only `2` and `sqrt(5)`, whose spacing is below `.25`. This makes
the locked start set empty even though `[1,2,2.75]`, qmax `[2,4,4]` is a
feasible continuous point. The oracle records the contradiction instead of
adding an unapproved seed. It cannot change the selected result because 7x7
adequacy is conditional on an adequate 9x9 high-order reference, which failed.

The fixed architecture uses equal F0/F1 **representation-copy** capacity at
each hidden stage:
one F0 copy and one two-component F1 copy each consume one logical slot. At the
baseline logical widths `[32,48,64,96]`, the fixed copy pairs are
`[(16,16),(24,24),(32,32),(48,48)]`, producing packed tensor widths
`[48,72,96,144]`. Exact analytic recounting with the selected profiles gives
`1,172,304` basis coefficients, `3,600` normalization parameters, `4,096` gate
parameters, and `35` scalar biases: `1,180,035` total learned parameters, below
the `3,958,435` cap. Dense convolution MACs are `159,837,585,408` per sample
and basis-expansion MACs are `159,453,168` per forward.

The targeted `--refresh-layout` path loaded the already selected profiles and
did not enter radial search. Its independently initialized 128-trial escnn
comparison covers 13 distinct layer signatures and 25 frequency outputs; every
mean variance ratio passes, spanning `0.9977233322` to `1.0079137704`. The
tracked model handoff contains only `7-low`, `9-low`, their fixed assignment,
and the equal-copy layout. Rejected profiles and the premise finding remain only
in the audit. Hashes of the audit's search, profile, escnn-reference, high-order,
and locked-premise sections are unchanged by the refresh.

Spec 0013's local architecture probe now implements and verifies fixed padded
`torch.bmm` hidden expansion, fixed `torch.mm` scalar-boundary expansion,
direct static block assembly, one dense `conv2d`,
field norm/gate AMP rules, compile guards, and focused escnn/CPU tests. V2
passes every numerical/runtime-correctness gate and establishes padded `bmm`
plus direct assembly as the fastest fixed path. The user replaced the isolated
10% assembly decomposition with per-block operational gates. The singular
four-path confirmation passed all numerical/runtime ratios. The user selected
compiled execution and made its cross-rank correlated timing CV diagnostic, so
Spec 0013 is accepted without a rerun. The full convolution topology and VAE
remain separately unauthorized.

## Non-Goals

- Runtime `escnn`, O(2), discrete rotations, arbitrary group/field layouts, or
  production-style compatibility/configuration layers.
- Mixed-irrep VAE latents; the first latent remains `16F0` at `32x32`.
- An equivariance loss, learned resampling, 1x1 layers, or a learned radial
  basis in the primary comparison.
- Exact continuous-rotation behavior on a finite pixel grid.

## One-Off Specialization Rule

The basis oracle is offline selection code and may enumerate only the four
declared profiles. After it selects one architecture, the executable
training/model path supports exactly that result. Do not turn the candidates
into a runtime configuration surface.

Concretely, the selected implementation has:

```text
one fixed field schedule
one fixed support assignment
one fixed set of radii, widths, and q masks
one fixed packed channel order
one fixed normalization/gate policy
one fixed scalar latent and RGB interface
```

The final model must not expose switches for `F012` versus `F01`, 7x7 versus
9x9 F2 support, alternative profiles, radial families, shell counts, group
types, arbitrary frequencies, arbitrary channel order, or learned/adaptive
radii. Rejected candidates remain only in the tracked numerical audit; they do
not remain as selectable model implementations.

Small private helpers may receive construction-time tensor shapes needed to
avoid duplicating the 43 convolution positions, but this is code reuse, not a
general API. They assume only the frozen `F0/F1/F2` layout and selected
manifest. There is no runtime field registry, profile lookup, fallback,
auto-tuning, shape-driven architecture selection, or support change. All
branches are resolved while constructing the one model, and every forward has
static tensor shapes and one dense `conv2d` per learned-convolution position.

Do not add compatibility layers, plugin hooks, malformed-config handling,
checkpoint migrations, generic serialization, or tests for configurations the
experiment will never run. Tests protect the selected experiment and the
mathematical invariants that made it selectable.

### Compile-Friendly Boundary

The offline oracle may use Python control flow, SciPy, SVD/QR, `escnn`, and
JSON because none of it runs during training. Once the manifest is selected,
the training model never invokes the oracle or reads/chooses profiles in
`forward`.

At model/module construction time, do all fixed work:

```text
load or hard-code the one selected manifest
resolve every FieldSpec offset and component slice
sample Gaussian radii and angular matrices
apply centre masks and the selected QR coordinates
construct the one-copy basis bank for each legal frequency pair
register those banks as fixed FP32 buffers
allocate coefficient tensors with their final static shapes
resolve bias permission, padding, and output block order
```

Do not repeat `sqrt/atan2/sin/cos`, Gaussian sampling, QR/SVD, rank checks,
manifest parsing, pair discovery, mask construction, field-layout inspection,
or basis normalization in `forward`.

The learned coefficients change after every optimizer step, so training must
still form the dense convolution kernel on each forward. That necessary linear
expansion is the only basis calculation left there. For each of the at most
nine fixed frequency pairs, contract

```text
coeff[n_out_copies, n_in_copies, basis_dim]
    x pair_basis[basis_dim, d_out, d_in, k, k]
    -> dense pair block
```

with a static `einsum`/matrix multiplication, concatenate the fixed blocks in
canonical channel order, then call exactly one dense `conv2d`. Implement the
selected pair contractions explicitly or as a fixed unrolled tuple; do not
discover pairs or loop over field copies at runtime. Empty frequencies are
resolved in `__init__`, not branched on in `forward`.

Conceptually:

```python
def forward(x):
    k00 = expand(c00, basis00)
    k01 = expand(c01, basis01)  # only pairs present in this fixed layer
    # fixed remaining pair expansions, then static cat/reshape
    kernel = assemble_fixed_blocks(k00, k01, ...)
    return conv2d(x, kernel, bias=fixed_allowed_bias, padding=fixed_padding)
```

The ellipsis describes source code generated/written for the selected fixed
pair set, not a dynamic container scan. `torch.compile` sees fixed buffer and
parameter shapes, tensor contractions, concatenations, and one convolution.
An eval-only expanded-kernel cache is optional and must not complicate or
branch the compiled training path.

## Fixed Comparison Contract

Keep the normal VAE's RGB input/range, spatial Gaussian latent, loss, beta
`0.01`, data/split protocol, eight encoder and eight decoder residual blocks,
43 learned-convolution positions, 34 activation positions, and the three
`256 -> 128 -> 64 -> 32` transitions (12 branch resampling operations). Keep
the branch ordering and projection locations. Baseline widths
`[32,48,64,96]` are logical representation-slot budgets, not required packed
widths. The selected F01 layout has packed widths `[48,72,96,144]`, uses a 9x9
stem and 7x7 at every other learned-convolution position, and remains within
the learned-parameter cap. This is one architecture, not a hyperparameter
sweep.

## Field Contract

For a rotation by `theta`, use `F0: rho_0=1` and `F1: rho_1=R(theta)`. The
offline oracle validates frequencies through F2, but the selected training
layout contains only F0 and F1. F1 is an ordinary two-component vector
representation.

Use one static NCHW layout in every layer:

```text
[F0 copies | F1 copy 0 (cos,sin) | ...]
```

Frozen stage `FieldSpec(n0, n1, n2)` constants own offsets and field-aligned
reshapes. No runtime `GeometricTensor`, dynamic field registry, per-copy
forward loop, or arbitrary-layout support is allowed. Once selected, code may
specialize offsets, basis buffers, and shapes to this one model/runtime.
RGB input, `mu`, `logvar`, sampled latent, and RGB output are scalar fields.
The selected F0/F1 model is valid continuous SO(2), with pair paths only
through q=2. Capacity is counted in representation copies: an F1 copy counts
as one logical slot even though it occupies two tensor components. Packed
physical width is therefore allowed to exceed the baseline width; learned
parameter count is the comparison constraint.

### Primary Field Schedule

Lock this equal-copy schedule for the first EQ run:

| resolution | F0 copies | F1 copies | packed channels |
| ---: | ---: | ---: | ---: |
| input | 3 | 0 | 3 |
| 256 | 16 | 16 | 48 |
| 128 | 24 | 24 | 72 |
| 64 | 32 | 32 | 96 |
| 32 | 48 | 48 | 144 |
| latent | 16 | 0 | 16 |

This preserves the control's logical number of representation slots, split
equally between scalar and vector copies, rather than its packed activation
width. It avoids charging a two-component F1 representation as two independent
baseline scalar features. This is a capacity convention, not a theorem or a
claim that the two field types have identical learned utility.

Residual branches may add only when their output `FieldSpec`, canonical layout,
component convention, and spatial grid are identical. A raw identity skip is
therefore legal only when the input and output `FieldSpec` are identical. If a
block changes any multiplicity or grid,
its skip is fieldwise resample if needed, steerable projection to the target
`FieldSpec`, field-aware normalization, then add. Equal flattened channel count
does not permit a raw identity. Keep one `FieldSpec` per resolution stage so
field changes occur only at the three encoder/decoder transitions; the stem and
latent projection are also steerable maps to their fixed stage specs.

## Steerable Convolution Contract

An `F_m -> F_n` kernel block obeys
`K(R_theta x) = rho_n(theta) K(x) rho_m(theta)^-1`. The field cap does **not**
cap spatial kernel order. Retain the full pair-derived paths:

| output / input | F0 | F1 | F2 |
| --- | ---: | ---: | ---: |
| F0 | q=0 | q=1 | q=2 |
| F1 | q=1 | q=0,2 | q=1,3 |
| F2 | q=2 | q=1,3 | q=0,4 |

`F1->F1` and `F2->F2` q=0 paths include both real SO(2) intertwiners `I` and
`J`; retaining only `I` would be an unnecessary O(2)-like restriction.

### Exact Real Basis Convention

Use kernel coordinates `x = column - centre` and `y = centre - row`, so `x`
points right and `y` points up. Let

```text
r = sqrt(x*x + y*y)
phi = atan2(y, x)
R_l(theta) = [[cos(l*theta), -sin(l*theta)],
              [sin(l*theta),  cos(l*theta)]]
```

The active field action is

```text
(T_theta f_l)(x) = R_l(theta) f_l(R_-theta x),
```

with `R_0 = 1`. For radial shell `g_j`, every sampled basis block is

```text
B[j,A,n<-m](r, phi) = g_j(r) R_n(phi) A R_m(-phi).
```

For scalar sides, `A` has the corresponding `2x1`, `1x2`, or `1x1` shape. Use

```text
I = [[1, 0], [0, 1]]       J = [[0, -1], [1, 0]]
S = [[1, 0], [0, -1]]      T = [[0,  1], [1, 0]].
```

The complete generator table is:

| pair | generators | spatial order |
| --- | --- | ---: |
| `F0 -> F0` | `[1]` | 0 |
| `F0 -> Fl` | `[1,0]^T`, `[0,1]^T` | l |
| `Fl -> F0` | `[1,0]`, `[0,1]` | l |
| `Fl -> Fl`, `l>0` | `I,J` and `S,T` | 0 and `2l` |
| `Fm -> Fn`, `m,n>0`, `m!=n` | `I,J` and `S,T` | `abs(n-m)` and `n+m` |

This table fixes signs and multiplicities. `torch.nn.functional.conv2d` uses
cross-correlation; sample the basis with the coordinates above and do not flip
it again. A 90-degree comparison against `escnn` must fail if the chosen image
rotation sign, component order, or cross-correlation convention is wrong.

The same static convolution implements lifting and projection:

```text
RGB lift:        3F0 -> 16F0 + 16F1
hidden maps:     fixed F0/F1 -> fixed F0/F1
latent heads:    48F0 + 48F1 -> 16F0
RGB projection:  16F0 + 16F1 -> 3F0
```

The selected final RGB projection is learned from the 256-stage
`16F0+16F1`, so it aggregates `F0 -> F0` and `F1 -> F0` (`q=1`) paths. The
scalar latent heads at the 32-stage directly aggregate all legal F0/F1-to-F0
paths. Projection is not an inverse of lifting and must not discard F1 fields
present at its input.

The one-off model keeps the control's external interface exactly: attributes
and methods `latent_channels=16`, `encode(inputs) -> (mu,logvar)`,
`decode(z) -> reconstruction`, `reparameterize(...)`, and
`forward(inputs, eps=None) -> VaeForwardOutput`. `mu`, `logvar`, `z`, and `eps`
remain scalar tensors of shape `(B,16,32,32)`. No generic model interface is
needed.

An additive bias is an invariant constant field, so it is permitted independently
per `F0` output copy and forbidden for `F1`/`F2`. Mirror the completed control's
placement and initialization rather than adding biases everywhere:

- EConvs followed immediately by field-aware normalization have no convolution
  bias. The normalization is the only hidden additive-shift location: its `F0`
  affine bias is initialized to zero and scale to one; nontrivial fields have
  scale only.
- Scalar `mu` and `logvar` heads have a learned scalar bias, as the control's
  unnormalized heads do. Their initialization follows ordinary Conv2d default
  initialization; it is not explicitly zero in the completed control.
- The final scalar RGB head has a learned bias, but its entire expanded kernel
  and scalar bias are initialized to zero, matching the control's zero-output
  start.

The learned `b` inside a scalar/radial gate is distinct from a field bias and
starts at zero for every field copy. Count all allowed scalar biases.

Precompute the selected tensor-valued basis as FP32 buffers. If `Q[p,o,i,u,v]`
is the stored basis and `c[p]` its coefficients, each layer performs

```python
kernel = einsum("p,poihw->oihw", coefficients, Q)
output = conv2d(input, kernel, padding=kernel_size // 2)
```

The actual implementation may flatten or batch the contraction, but it must
produce one dense convolution kernel without a forward loop over field copies.
Evaluation may cache the expanded kernel. Coefficients are independent per
input/output field-copy pair; the implementation must not silently tie them.

The centre is a dedicated fixed q=0 impulse/intertwiner basis. All q>0 basis
samples are exactly zero there. This does **not** force every nontrivial-field
centre coefficient to zero: `F1->F1` and `F2->F2` retain their legal q=0
`aI+bJ` centre maps.

## Bounded Radial And Support Search

Radii are not learned in the first model. For a non-centre shell,

```text
g_j(r) = exp(-(r-r_j)^2 / (2 sigma_j^2)).
```

The origin is a separate exact impulse, not a learned or searched Gaussian.
For the `escnn` comparison, represent that impulse by its conventional
`r=0,sigma=0.005` shell and first assert in FP64 that every sampled non-origin
value is below `1e-12`; replace that sampled column by the exact impulse before
the span comparison.

Radii are continuous; they do not need to coincide with integer or diagonal
pixel distances. Use the following grid only to produce starting points:

| support | number of non-origin shells | allowed centres | allowed widths |
| --- | ---: | --- | --- |
| 7x7 | 3 | choose an increasing subset of `{1,sqrt(2),2,sqrt(5),sqrt(8),3}` | common inner sigma plus outer sigma, each in `{0.4,0.5,0.6,0.7}` |
| 9x9 | 4 | choose an increasing subset of `{1,sqrt(2),2,sqrt(5),sqrt(8),3,sqrt(10),sqrt(13),4}` | common inner sigma plus outer sigma, each in `{0.4,0.5,0.6,0.7}` |

For each centre list, enumerate nondecreasing integer
`qmax_j in {0,1,2,3,4}` values. Audit four profiles:

```text
7-low and 9-low:   used q = {1,2}
7-full and 9-full: used q = {1,2,3,4}
```

All four profiles also retain every legal q=0 radial/intertwiner path.

Each profile satisfies

```text
qmax_j <= min(4, floor(2*r_j)),
at least two non-origin shells retain every q used by that profile,
and the last shell retains the profile's largest used q.
```

The discrete q mask stays fixed during refinement. For every coarse candidate,
compute the ordered maximization key

```text
(number of full-rank pair bases,
 sum of the nine pair ranks,
 total retained basis dimension,
 -worst finite pair kappa,
 worst pair minimum singular value).
```

Retain the best 16 starts per profile; exact ties use the lexicographically
smallest `(centres,widths,qmax)`. Refine each with SciPy 1.18 COBYQA. This is a
deterministic, derivative-free constrained solver suited to the six or eight
real variables here; no image data, gradients through rank tests, or random
starts are used. The variables are

```text
x = (r_0,...,r_(J-1), sigma_0,...,sigma_(J-1)).
```

Give COBYQA direct `Bounds` and one `LinearConstraint` encoding

```text
R = (kernel_size - 1) / 2
max(0.25, qmax_j/2) <= r_j <= R - 0.25
r_(j+1) - r_j >= 0.25
0.30 <= sigma_j <= 0.90.
```

The axis radius `R`, rather than the corner radius `sqrt(2)*R`, is the upper
bound because a corner-centred ring has too few angular samples for q=3/q=4.
Before launching a start, solve the simple bound/spacing feasibility check; if
no feasible ordered radii exist, discard that mask. Discard any coarse start
that itself violates the bounds or `0.25` spacing rather than silently moving
it before its first objective evaluation. Use

```text
method="COBYQA"
maxfev=4000
initial_tr_radius=0.1
final_tr_radius=1e-7
feasibility_tol=1e-10
scale=True
```

and otherwise SciPy 1.18 defaults. Evaluate the initial and final point even if
the solver exhausts `maxfev`. A final point is feasible only when every bound
and linear-constraint violation is `<=1e-10`; if it is non-finite or violates
that limit, reject it and retain the initial point. Record the solver status.
The hard numerical gates below, not the solver's success flag, decide whether
the result can enter the final comparison.

For each retained field-pair matrix with unit-normalized columns, define

```text
G_p = B_n,p^T B_n,p
ell_p = -logdet(G_p + 1e-8 I) / number_of_columns(G_p)
L = 0.05 * logsumexp(ell_p / 0.05 over retained pairs).
```

Minimize `L`. This smooth objective improves the worst nearly dependent pair;
it does not itself accept a basis. After refinement, recompute rank, singular
values, condition number, support norms, and the `escnn` checks from scratch.
If optimization worsens the hard score below, retain that start's unrefined
version. Final selection considers all passing coarse candidates plus the 16
refined candidates. This is a deterministic numerical basis search, not
learning radii from images or reconstruction loss.

For each field pair, flatten the raw sampled basis to a matrix `B` whose columns
are basis elements. Discard a candidate if any column norm is below `1e-10`.
Normalize each column to unit L2 norm, call the result `B_n`, and compute

```text
s = svdvals(B_n)
rank = count(s > 1e-10 * s[0])
kappa = s[0] / s[-1]
```

A candidate passes only when every retained pair basis has full column rank and
`kappa <= 10`. Before storing it, compute a reduced Euclidean-orthonormal QR
basis `Q_unit`; fix each column sign by making its largest-magnitude entry
positive. For an output irrep of real dimension `d_n` (`1` for F0, `2` for
F1/F2), store `Q = sqrt(d_n) * Q_unit`. This matches `escnn`'s sampled-basis
norm convention while preserving the span. QR happens after the raw
rank/condition audit, so whitening cannot hide a bad candidate.

For comparisons only, round `kappa` and minimum singular values to 10 decimal
places and centres/widths to 8 decimal places; store those rounded
centres/widths in the manifest and rebuild the final audit from them. Among
passing candidates for one profile, select lexicographically:

1. largest total retained basis dimension across the nine field pairs;
2. smallest worst-pair `kappa`;
3. largest worst-pair minimum singular value;
4. lexicographically smallest `(centres, widths, qmax)` tuple.

The oracle writes one result for each of `7-low`, `7-full`, `9-low`, and
`9-full`, with selected values, pairwise dimensions, singular values,
condition numbers, and architecture parameter counts. A full profile may fail
without preventing selection of its low profile. These selected results,
rather than this candidate grid, become the fixed model inputs after the oracle
runs.

### Shared Profiles, Not Per-Layer Search

The four profiles are the complete radial search surface. Each profile selects
one global `(radii, widths, qmax_per_shell)` tuple shared by every convolution
assigned to it. Do **not** rerun COBYQA or choose different radial parameters
for encoder versus decoder, resolution, block number, main versus skip branch,
latent head, RGB head, or particular input/output multiplicities.

Once a profile is fixed, a concrete layer merely constructs the legal subset
for its `FieldSpec_in -> FieldSpec_out` and repeats the same one-copy pair basis
over all input/output copies with independent learned coefficients. Different
layer shapes therefore produce different-sized basis buffers and coefficient
tensors, but not newly optimized radii, widths, q masks, or supports.

The offline oracle evaluated these static assignments:

```text
candidate A: stem=9-low; every other convolution=7-full
candidate B: stem=9-low; F2-bearing convolutions=9-full; all others=7-low
F0/F1:       stem=9-low; every other convolution=7-low
```

Here “F2-bearing” means `FieldSpec_in.n2>0 or FieldSpec_out.n2>0`. Legal pair
orders still come from the frequency table: assigning `7-full` to a layer with
no F2 does not invent q=3/q=4 paths. The selected assignment is global and is
not varied by layer position. The measured result is F0/F1 only: `9-low` at
the stem and `7-low` everywhere else. Candidates A and B are rejected evidence,
not implementation branches.

There is no search over alternative radial families, numbers of shells,
normalizations, nonlinearities, field proportions, initialization families,
solvers, or individual layer supports in this slice. COBYQA is the fixed
numerical refiner, not another bakeoff. Coefficient initialization is
calculated from the chosen basis and layer multiplicities; it is verified, not
optimized. After the manifest is written, all search stops and the specialized
equivariant convolution is implemented from that manifest.

The one-off slice owns only these artifacts:

```text
src/eqvae/models/so2_basis.py                 analytic sampled basis
scripts/select_so2_basis.py                   finite search and count oracle
configs/spec0012/so2_basis_manifest.json      selected model input
docs/data/spec0012_so2_basis_audit.json       tracked numerical evidence
tests/test_so2_basis.py                       algebra/reference regressions
```

The audit records the `escnn` source URL and commit; the current reference is
`QUVA-Lab/escnn@9ad44cc37d69`. A local checkout or test environment is allowed
for the oracle, but neither `escnn` nor its generic tensors enter the selected
model or Kaggle training payload. The current project venv does not install
`escnn`. The oracle may prepend the ignored `reference/escnn` checkout to
`sys.path` and use two test-only import shims: a no-cache replacement for
`joblib.Memory`, and a `lie_learn` SO(3) sentinel that raises if the SO(2) oracle
accidentally enters an SO(3) path. Existing Torch/NumPy/SciPy are sufficient;
do not add `pymanopt`, `autograd`, `py3nj`, or a Fortran toolchain for this
check. The bootstrap must be private to the oracle/tests and must never modify
the process-wide training entry point or ordinary `PYTHONPATH`.

The layout-only refresh and reproduction commands are

```bash
.venv/bin/python scripts/select_so2_basis.py --refresh-layout --write
.venv/bin/python scripts/select_so2_basis.py --refresh-layout --check
.venv/bin/python -m pytest -q tests/test_so2_basis.py
```

They load the selected profiles and never enter the four-profile search. Plain
`--write/--check` remain full radial-oracle operations and are not part of a
layout refresh.

Do not learn radii or widths in the primary model. Any global radial function
of distance remains algebraically SO(2)-compatible, but learnable shells can
collide, lose sampled support, worsen conditioning, and force basis
reconstruction every training forward. A later ablation must keep the
shell-to-q mask fixed, pin the centre shell, constrain ordered non-overlapping
centre bands and positive bounded widths, construct the basis in FP32 outside
autocast, and count all radial parameters.

## Normalization And Nonlinearities

Keep all baseline normalization/gate locations and use `eps=1e-5`. For `F0`,
use eight groups and ordinary per-sample GroupNorm over copies and spatial
positions:

```text
y_i = gamma_i (x_i - mean_group) / sqrt(var_group + eps) + beta_i.
```

For `F1` and `F2`, use four groups per frequency. If `G(i)` is the group of
same-frequency copies containing copy `i`, compute no mean and use

```text
rms_G = sqrt(mean_{copy in G, component, h, w}(x^2) + eps)
y_i = gamma_i x_i / rms_G.
```

There is one `gamma_i` per field copy, shared by its two components, and no
nontrivial-field `beta`. Initialize every `gamma=1` and every allowed
`F0 beta=0`. Compute statistics in FP32 under AMP and cast the normalized result
back to the surrounding dtype. The selected schedules are divisible by these
group counts; a missing frequency is skipped rather than assigned an empty
group.

`F0` uses the shared scalar gate. Every nontrivial copy uses
`r=sqrt(u^2+w^2+eps)`, `sigmoid(a*r+b)*[u,w]`, with one `a,b` pair per copy,
`eps=1e-4` initially, zero gate weight decay, and gate LR multiplier 0.5.

## Coefficient Initialization

Use the specialized form of `escnn` generalized-He initialization. For output
field copy `b`, let `T_b` be the number of input frequencies present in the
layer and let `D_(m,b)` be the total number of retained coefficients from all
input copies of frequency `m` into `b`. Every such coefficient is initialized
independently as

```text
z ~ Normal(0, 1)
c = z / sqrt(T_b * D_(m,b)).
```

This is computed from the final QR basis manifest and mirrors the reference
initializer without importing `escnn` at training time.

Hidden normalized convolutions and the unnormalized `mu`/`logvar` heads use
this coefficient initializer. Their scalar head biases retain the completed
control's PyTorch Conv2d default uniform initialization with physical
`fan_in = C_in * kernel_size^2`. The final RGB head sets every coefficient and
its scalar bias to zero. Norm and gate initialization remains `gamma=1`,
`beta=0`, `a=1`, `b=0`.

The oracle must run 128 fixed-seed unit-variance trials for each distinct layer
type and compare output variance by frequency with an identically configured
`escnn` layer. The mean variance ratio must lie in `[0.9,1.1]`; this checks the
formula but does not tune initialization from training loss.

## Resampling Contract And Decision Gate

Resampling is fieldwise: one scalar spatial operator applied identically to
every packed component (`S tensor-product I_fiber`), without field reordering,
mixing, or learned parameters. Keep bilinear uniform x2 with
`align_corners=False` on both decoder branches; its exact 90-degree primitive
check passes on the even grid.

The trained control currently uses binomial 5x5 blur plus stride-2 decimation,
implemented as a fixed grouped convolution; it does not use a learned stride-2
convolution. Its even-grid sampling phase fails the required 90-degree primitive
check: decimation selects one pixel-parity coset, while a 90-degree rotation
about the half-pixel image centre maps it to the other. The blur itself remains
rotation-compatible; this is a lattice-geometry defect, not a vector-field or
floating-point defect. Lock the primitive test before attaching a numerical
error value to this claim.

The optional strict-equivariance variant is a fixed, fieldwise, phase-centred 6x6 separable filter
`outer([1,5,10,10,5,1]/32, same)` with `stride=2,padding=2`. In FP64 it is the
same operator as 5x5 binomial blur at stride one followed by half-scale bilinear
resize with `align_corners=False`; it preserves the one-resampling-op topology
and exact 90-degree C4 compatibility on the even grid. It costs 36 rather than
25 fixed MACs per output component, a 44% increase only in the small fixed
resampling budget. Non-cardinal SO(2) errors still require measurement. It is
not numerically the same operator as the completed control, so it requires a
new matched normal-VAE control and is not in the first comparison.

Primary comparison choice: retain the completed control's fixed 5x5 binomial
blur plus stride-2 decimation, applied fieldwise to the EQ canonical packed
tensor. This removes an avoidable resampling-operator confound and avoids
retraining the completed normal control. Its known sampled-grid phase error is
an accepted limitation: measure and report it at primitive, transition, and
full-model levels. The phase-centred 6x6 operator remains documented only for a
later matched rerun, not as an EQ-only modification.

All resampling claims remain sampled-grid approximations. Test F0/F1/F2
primitive, branch, transition, and full-model errors at 90 degrees and
non-cardinal angles, reporting full-frame and border-cropped errors alongside
the image-transform round-trip floor.

## Capacity And Runtime Selection

The small architecture-selection oracle is the source of truth; it is not part
of training. For each candidate it reports sampled-basis rank per pair/ring,
scalar biases, normalization/gate parameters, total learned parameters,
dense-convolution MACs, basis-expansion cost, and physical tensor widths. The
preferred 9x9/7x7 candidate is expected to remain far below the parameter cap,
so matching tensor widths is not a parameter-matched comparison.

### `escnn` Correctness Oracle

`escnn` is test-only reference code, not a training dependency. For every
one-copy `F_m -> F_n` pair, construct its basis with the same support, centres,
widths, and per-radius cutoff. If `Q_ours` and `Q_ref` are FP64 orthonormal
bases of the flattened sampled kernels, require equal dimensions and

```text
d_span = ||Q_ours Q_ours^T - Q_ref Q_ref^T||_2 <= 5e-5.
```

Raw basis columns need not have the same order or signs. Generate a dense
kernel from our basis, project it into the reference span, and require relative
kernel residual `<=5e-5` and FP32 convolution-output relative RMS `<=1e-4`.
These tight checks detect wrong signs, component order, missing `I/J`, or a
wrong cross-correlation convention.

For sampled rotations, use the same `escnn` field transform, input, crop, and
kernel for both implementations. On deterministic smooth `65x65` fields, test
`15,30,45,60,90` degrees and compute

```text
E(theta) = ||crop(C(T_theta x) - T_theta C(x))||_2
           / max(||crop(T_theta C(x))||_2, 1e-8).
```

Use exact `rot90` at 90 degrees and the reference bilinear transformation at
the other angles. Crop `kernel_size//2 + 4` pixels. Our implementation passes
when

```text
E_ours(theta) <= max(5e-4, 1.10 * E_escnn(theta))
```

for every pair and angle. The small margin acknowledges implementation and
sampling differences; it does not relax the tighter kernel-span check. The
known 5x5 blur-decimation phase error is measured separately and cannot reject
an otherwise correct F2 kernel basis.

### Architecture Decision Ladder

The only pre-training search was numerical and deterministic. The basis-oracle
slice selected one candidate through this locked decision ladder:

1. Select and audit `7-low`, `7-full`, `9-low`, and `9-full` independently.
   Failure of a full profile does not remove its low profile.
2. Measure F2 representation power and brittleness. For a full profile, define

   ```text
   B_low,p  = sampled columns for q<=2 in F2-bearing pair p
   B_full,p = sampled columns for every legal q in pair p
   D_high(k) = sum_p [rank(B_full,p) - rank(B_low,p)]
               for p in {F1->F2,F2->F1,F2->F2}, using the ordinary
               1e-10*s_max rank tolerance.
   E_floor = maximum image-transform round-trip error on the same inputs,
             angles, interpolation rule, and crop.
   ```

   This is the incremental sampled-kernel freedom added by q=3/q=4 after
   accounting for overlap with q<=2. A profile with `D_high=0` has gained no
   usable F2 high-order representation power even if its raw high-q columns
   are individually nonzero.

   For each profile's selected nominal manifest, form every simultaneous
   perturbation vector

   ```text
   delta in {-0.02,0,+0.02}^(2J), delta != 0,
   ```

   over all `J` radii and `J` widths. Keep only perturbations that already obey
   the same bounds and spacing constraints—do not project them—then round to 10
   decimal places, remove duplicates and the nominal point, and audit every
   remaining point with the q mask fixed. Require at least one retained
   perturbation that changes each radius and each width; otherwise the profile
   is not robust. A full profile is robust only if every retained perturbation
   remains full rank with worst-pair `kappa<=12` and `D_high` at least its
   nominal value. The nominal manifest still requires `kappa<=10`.

   Define `E_high(k)` intrinsically on the incremental high-order subspace, not
   on arbitrary raw Gaussian columns. For each F2-bearing pair, project the
   q=3/q=4 columns orthogonally away from `span(B_low,p)`, then take an SVD with
   the same rank tolerance to obtain a canonical orthonormal incremental basis
   `H_p`. Use an FP64 input bank of eight `65x65` tensors per input irrep:
   independent `Normal(0,1)` values from a local generator seeded `12012`,
   blurred by a normalized separable Gaussian with `sigma=2`, radius 6, and
   normalized to unit RMS per tensor. A two-component input uses independent
   generated components.

   For every angle, let column `j` of `A_theta` be the cropped equivariance
   residual produced by incremental kernel `H_p[j]`, flattened and stacked over
   the eight inputs. Let column `j` of `Y_theta` be the corresponding cropped
   rotated-reference output. Define the worst linear combination by

   ```text
   E_subspace(theta)^2 = largest generalized eigenvalue of
       A_theta^T A_theta v = lambda (Y_theta^T Y_theta + 1e-8 I) v
   E_high(k) = max over pairs and angles {15,30,45,60,90}
                     sqrt(E_subspace(theta)^2).
   ```

   This value is unchanged by a different orthonormal basis of the same
   incremental subspace. Verify the implementation against the matched
   `escnn` span before using the value.

   Let `E_limit=max(0.05,1.5*E_floor)`. The 9x9 F2 basis is the required
   high-order reference: if it is not nominally full rank, has `D_high(9)=0`,
   is not robust, or has `E_high(9)>E_limit`, neither support is adequate and
   the decision routes to F0/F1. A 7x7 F2 basis is adequate only when 9x9 is
   adequate and 7x7 is robust,

   ```text
   D_high(7) / D_high(9) >= 0.75
   E_high(7) <= E_limit
   ```

   and

   ```text
   E_high(7) - E_high(9) <= max(5e-4, 0.25 * E_high(9)).
   ```

   Thus 7x7 may lose at most 25% of the available stable high-order basis
   dimension and may not be materially less faithful than 9x9. These are
   representation/sampling gates, not trained-quality comparisons.
3. Candidate A uses `9-low` for the stem and `7-full` elsewhere with the
   primary `F0/F1/F2` schedule. Select it only if the 7x7 F2 basis is adequate
   and its analytic learned-parameter count is `<=3,958,435`.
4. Otherwise, candidate B uses `9-full` at every convolution whose input or
   output contains F2, `7-low` at all other non-stem positions, and `9-low` at
   the stem. Select it only if the 9x9 F2 basis is adequate and its analytic
   learned-parameter count is within the same cap.
5. If neither full profile is adequate, or candidate B exceeds the parameter
   cap, select F0/F1. After the oracle selected that route, the user locked
   equal F0/F1 representation-copy capacity:

   | resolution | F0 copies | F1 copies | physical channels |
   | ---: | ---: | ---: | ---: |
   | 256 | 16 | 16 | 48 |
   | 128 | 24 | 24 | 72 |
   | 64 | 32 | 32 | 96 |
   | 32 | 48 | 48 | 144 |

   It uses `9-low` at the stem, `7-low` elsewhere, and only q=0,1,2 paths. F2
   failed its prerequisite gate, so this is the sole selected field layout.

If a q<=2 path fails the 7x7 reference gate, treat that as a basis bug first;
do not hide it by changing the architecture. Numerical tests can decide whether
F2 is represented correctly, retains enough high-order kernel freedom, is not
brittle to small radial changes, and is computationally feasible. They cannot
establish whether F2 improves reconstruction quality without training, so no
claim of task-optimal field proportions is permitted.

The separately reviewed Spec 0013 architecture-probe slice implemented the
selected convolution mechanics, normalization, gates, one residual block, and
one encoder/decoder transition. Its compiled dual-T4 forward/backward proof at
per-device batch 4 is accepted. Full-VAE coding is mechanically ready but
remains a new, separately authorized implementation slice. Later measured
full-model runtime may trigger only a narrow Spec 0011 follow-up; it must not
silently change the field schedule.

No training search over F0/F1 proportions, width multipliers, or radial
hyperparameters is planned. The equal-copy F01 layout is the single selected
capacity convention.

## Acceptance Criteria Before Full-VAE Coding

1. Complete: the small basis oracle writes all four profile outcomes and raw
   rank/condition/reference evidence without implementing or training the VAE.
2. Complete: the oracle selects F01 and passes the independent one-copy
   `escnn` pair-kernel tests.
3. Complete: the targeted layout refresh reproduces
   `1,180,035 <= 3,958,435`; all 25 refreshed initialization ratios pass without
   repeating radial search or changing its evidence.
4. Complete: Spec 0013 locks RGB lifting/projection, field layout, residual
   compatibility, normalization, radial gates, resampling, contraction and
   assembly mechanics, and compiled dual-T4 acceptance limits.
5. Complete: the singular padded-`bmm` implementation passes focused local
   correctness and the accepted remote compiled-performance contract. The
   baseline downsample phase error is reported rather than misclassified as a
   kernel failure. Full-model runtime selection remains later Spec 0011 work.

## Related Files

- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0013-fixed-f01-architecture-probe.md`
- `docs/decisions/0004-so2-gaussian-ring-kernel-basis.md`
- `docs/decisions/0005-gated-activation-policy.md`
- `docs/equivariant_vae_transition_plan.md`
- `src/eqvae/models/non_equivariant_vae.py`
- `src/eqvae/models/resampling.py`
