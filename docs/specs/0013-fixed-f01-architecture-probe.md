# Spec 0013: Fixed F01 Architecture Probe

Status: final dual-T4 probe failed timing-CV gate / explicit decision required
Owner/workstream: one-off continuous-`SO(2)` architecture probe
Last updated: 2026-08-13

## Purpose

Prove one compile-friendly implementation for the selected equal-copy F01
architecture before the full equivariant VAE is coded. This is a gate for one
experiment, not a reusable equivariance library.

Spec 0012 already selected `9-low` for the stem, `7-low` everywhere else, and
the fixed hidden copy pairs `(16,16)`, `(24,24)`, `(32,32)`, `(48,48)`. This
spec does not reopen that result. It fixes how those coefficients become one
dense kernel, how fixed fields are packed, and how the surrounding operations
behave under AMP and `torch.compile`.

Prefer the fastest simple implementation for these exact layouts, supports,
Torch runtime, dual-T4 hardware, and fixed shapes—even when that specialization
would be unsuitable as a general layer. Do not add abstractions, fallbacks,
dynamic options, generalized shape handling, or portability machinery unless a
measured acceptance failure requires the smallest such change. Bitwise equality
with escnn, eager FP32, or transformed finite-grid outputs is not a goal. The
explicit relative-error and sampled-equivariance limits below are the
correctness contract; ordinary FP16/compile rounding and the documented
resampling floor are acceptable within those limits.

The count refresh exposes an important cost premise: `1,180,035` learned
parameters are only 29.81% of the baseline cap, but the packed-width 9x9/7x7
dense convolutions cost `159,837,585,408` MAC/sample, 4.383x the baseline's
learned-convolution MACs. Parameter compliance is not compute matching. The
probe must measure this implementation honestly; it must not change the locked
layout to hide the cost.

## Implementation Status

The fixed local probe is implemented in
`src/eqvae/models/so2_architecture_probe.py`. It contains only the three
specialized convolution map kinds, fixed field norm/gates and resampling, one
identity block, one encoder transition, one decoder transition, RGB lift/head,
and scalar latent heads. It does not assemble or expose the full VAE. The
focused CPU evidence is tracked in
`docs/data/spec0013_so2_cpu_probe.json`: all 69 tests pass, every one-copy and
multi-copy escnn comparison passes, all 40 selected pair/angle rows pass the
escnn-relative sampled-equivariance rule, and the exact eventual count remains
`1,180,035`.

The one-use dual-T4 runner is locally built and guarded at
`kaggle/kernels/so2_architecture_probe`. It reads and hashes the exact selected
Spec 0011 runtime plan, verifies compiler/DDP readbacks, measures the four fixed
signatures against matched controls, and makes all accuracy, AMP, finiteness,
timing-CV, latency, graph, and VRAM limits load-bearing. Versions 1 and 2 are
complete. On 2026-08-13 the user accepted the evidence-based contract revision
below and selected padded `bmm` plus direct assembly for one final four-path
confirmation. Full-VAE assembly remains separately unauthorized.

### Dual-T4 v3 final result

Private Kaggle kernel version 3 ran from clean commit `c823a7e` on 2026-08-13.
Its tracked summary is `docs/data/spec0013_so2_dual_t4_probe_v3.json`. It passed
every numerical and operational gate except timing CV: maximum output and
coefficient-gradient relative RMS were `0.000619` and `0.000662`, all
compiled/eager ratios were `0.385..0.721`, all EQ/normal ratios were
`1.118..2.014`, parameters matched exactly across ranks, AMP/graph/recompile
counts were zero, and peak allocated/reserved memory was `798/954 MiB`.

The 10% timing-CV gate failed in 26 rank/block/path summaries. The pattern was
nearly identical across ranks: encoder window 0 and its pools failed for all
three modes, D-to-D window 1 and its pools failed for all three modes, and the
decoder normal-control pool narrowly failed. Window CVs ranged up to `0.598`;
pooled CVs ranged up to `0.538`. This shared-mode, shared-rank pattern is
consistent with environmental timing disturbance, but the predeclared rule is
still a failure. The D-to-D assembly fractions were diagnostic-only at
`0.387/0.374`.

The final-run stop rule is active. Do not rerun, add a mechanics arm, change a
runtime axis or tolerance, or assemble the full VAE without a new explicit
user/spec decision.

### Dual-T4 v1 result and fixed follow-up

Private Kaggle kernel version 1 ran from clean commit `e57f086` on 2026-08-13.
Its compact tracked summary is
`docs/data/spec0013_so2_dual_t4_probe_v1.json`. Runtime transfer itself was
healthy: 32 settled DDP updates completed with zero AMP skips, nonfinite values,
graph breaks, or post-settle recompiles; persistent buffers matched across
ranks; peak allocated/reserved memory was only `797/950 MiB`; compiled block
latency was `0.43..0.92x` eager and `1.02..2.04x` matched normal controls; the
topology-weighted ratio was `1.828`.

The current mechanics did not pass. The credible failure is the compiled FP16
`D -> D` coefficient-expansion-plus-assembly fraction: `0.407/0.413` across
ranks versus the unchanged `0.10` limit. V1 also exposed two measurement-path
defects that must be corrected, not waived: the decoder gradient diagnostic
omitted the selected GradScaler/compiled-autograd path, and the short D-to-D
eager/control timing body performed host scalar synchronizations. The observed
decoder gradient `0.05019` and eager/control CV failures are therefore not
accepted as model/runtime failures.

Exactly one targeted follow-up is predeclared before another remote write:

1. Make the accuracy diagnostic use selected FP16 autocast, GradScaler,
   compiled autograd, and unscaled FP32 master gradients. Retain the `2e-2`
   limit and record the worst parameter name, reference RMS, and difference
   RMS.
2. Move GradScaler and finiteness scalar reads outside timed intervals. For the
   short D-to-D row use 20 settled warmups and two untrimmed 50-sample windows,
   recording every sample; require each window and their pool to keep CV
   `<=10%`, and retain the compiled/eager and equivariant/normal limits.
3. Compare only three fixed D-to-D contraction/assembly arms: current four
   `mm` plus three cats; the same four `mm` plus one fresh final buffer with four
   non-overlapping fixed slice writes; and one padded four-bank `bmm` plus the
   same direct final assembly. Reverse arm order in the second timing window.
4. Keep coefficients, bases, layouts, parameter count, FP16 policy, gradient
   and output limits, graph/VRAM gates, and the `10%` assembly limit unchanged.
   Require the assembly fraction in each timing window and their pool; bind the
   candidate buffers, graph/recompile counters, and VRAM measurements to the
   candidate run itself.
   Select the lowest worst-rank compiled median only among fully passing arms.
   If none passes, stop; do not add another arm or runtime axis.
5. After a candidate passes this isolated comparison, make it the singular
   fixed implementation and run the complete four-signature acceptance probe.

This is an architecture-specific mechanics comparison forced by measurement,
not a runtime tuner. It does not reopen profiles, multiplicities, F2, or the
full-VAE scope.

### Dual-T4 v2 result and stop decision

Private Kaggle version 2 ran from clean commit `afec7af` on 2026-08-13. Its
tracked summary is `docs/data/spec0013_so2_dual_t4_probe_v2.json`. Corrected
accuracy passes comfortably: worst output relative RMS is `0.00061934` against
`0.005`, worst coefficient-gradient relative RMS is `0.00066145` against
`0.02`, and the formerly invalid decoder result is now `0.000554/0.000553`
across ranks. There are no missing/nonfinite gradients. Corrected D-to-D
compiled/eager is `0.817/0.813`, EQ/normal is `1.210/1.120`, all corrected-
control CVs pass, and DDP/AMP/compile/buffer/VRAM evidence is healthy.

No mechanics arm meets the unchanged assembly-fraction limit:

- four `mm` plus three cats: pooled `0.5042/0.5162`;
- four `mm` plus direct buffer: pooled `0.5025/0.5175`;
- padded `bmm` plus direct buffer: pooled `0.4465/0.4534`.

All per-window fractions lie in `0.4024..0.5630`, versus `0.10`. Independent
review recomputed the raw samples exactly and found that even each arm's most
favorable minimum-expansion/maximum-complete combination remains
`0.3107..0.3524`. Padded `bmm` is the fastest arm, at a `1.2655 ms` worst-rank
complete-forward median, but is not selectable under the contract. Narrow
pooled-CV failures do not affect this conclusion because every arm fails every
rank/window assembly gate independently.

The predeclared follow-up is exhausted. No fourth arm or runtime axis is useful.
The user explicitly chose the experiment-level alternative on 2026-08-13:
remove the isolated 10% fraction as a selection gate, lock padded `bmm` plus
direct assembly, and run one singular four-path confirmation. Kernel expansion
remains an unavoidable differentiable training cost because coefficients change
after every optimizer step. V2 already shows that the selected path is 10.5%
faster than the original D-to-D forward (`1.2655 ms` versus `1.4143 ms`) while
all numerical/runtime-correctness checks pass.

The old `1.828x` "topology-weighted" value is retained only in the v1 evidence,
not reused as a final gate. Its four composite-block weights do not map exactly
onto the 43 convolution positions: a measured residual block contains multiple
convolutions, transition signatures occur at different spatial resolutions,
and scalar stem/latent/head paths are absent. The final mechanics probe instead
requires every measured composite independently to pass its matched normal
control. Actual whole-model steps/second, peak memory, and projected epoch time
remain Spec 0011 measurements after separate full-model authorization.

## Non-Goals

- No radial search, F2 reconsideration, per-layer profile selection, learned
  basis, or alternate support.
- No arbitrary irreps, layouts, kernels, groups, manifests, or public config
  surface.
- No runtime `escnn`, QR/SVD, polar sampling, pair discovery, or basis legality
  logic.
- No full encoder, decoder, VAE, training run, checkpoint migration, or runtime
  search platform.
- No equivariance regularizer, O(2), reflection path, or new downsampling rule.
- No bitwise-reference parity, cross-hardware generality, or deterministic-mode
  work that is not required by an explicit acceptance check.

## Locked Inputs

- Manifest: `configs/spec0012/so2_basis_manifest.json`.
- Audit: `docs/data/spec0012_so2_basis_audit.json`.
- Baseline topology: `src/eqvae/models/non_equivariant_vae.py`.
- Input/output: scalar RGB in NCHW, normalized to `[-1,1]`.
- Latent: scalar `(B,16,32,32)` for `mu`, `logvar`, `z`, and `eps`.
- Primary downsample: existing fieldwise 5x5 binomial blur plus stride-2
  decimation, including its accepted even-grid phase limitation.
- Upsample: fieldwise bilinear x2 with `align_corners=False`.

The implementation hard-codes these layouts:

| symbol | role/resolution | copies `(F0,F1)` | packed channels | F1 offset |
| --- | --- | ---: | ---: | ---: |
| `R` | RGB | `(3,0)` | 3 | 3 |
| `A` | 256 | `(16,16)` | 48 | 16 |
| `B` | 128 | `(24,24)` | 72 | 24 |
| `C` | 64 | `(32,32)` | 96 | 32 |
| `D` | 32 | `(48,48)` | 144 | 48 |
| `L` | latent | `(16,0)` | 16 | 16 |

Packed order is `[all F0 copies | F1 copy-major (cos,sin) pairs]`. An F1 slice
is viewed as `(B,n1,2,H,W)` without changing the component order. There is no
alternate ordering or runtime `FieldSpec` parser.

## Offline And Construction Boundary

The checked-in manifest is read only by the construction helper or replaced by
equivalent hard-coded constants. `forward` never parses JSON. During offline
generation or module `__init__`, and nowhere later:

1. sample the locked Gaussian/angular columns;
2. apply centre legality masks and the selected QR coordinates;
3. flatten the four F01 pair banks;
4. resolve copy counts, offsets, coefficient shapes, padding, and bias policy;
5. register final contiguous FP32 basis buffers;
6. allocate the final FP32 coefficient parameters.

The training package contains no SciPy, NumPy, `escnn`, `joblib` shim, search
code, trigonometric coordinate generation, or manifest choice in its forward
path. The probe may use `escnn` only in focused tests.

## Pair Banks And Coefficient Shapes

Use `P_ab` for output frequency `a`, input frequency `b`. Store each pair bank
as a contiguous matrix `[p, d_out*d_in*k*k]`. Store its coefficients as
`[n_out*n_in,p]`; each row belongs to one output-copy/input-copy pair.

| profile | `P00` | `P10` | `P01` | `P11` |
| --- | ---: | ---: | ---: | ---: |
| `7-low`, k=7 | `[4,49]` | `[6,98]` | `[6,98]` | `[14,196]` |
| `9-low`, k=9 | `[5,81]` | `[8,162]` | `[8,162]` | `[18,324]` |

The fixed convolution signatures and coefficient row counts are:

| map | k | active coefficient matrices `[rows,p]` |
| --- | ---: | --- |
| `R -> A` | 9 | `C00[48,5]`, `C10[48,8]` |
| `A -> A` | 7 | `C00[256,4]`, `C10[256,6]`, `C01[256,6]`, `C11[256,14]` |
| `A -> B` / `B -> A` | 7 | four matrices with 384 rows |
| `B -> B` | 7 | four matrices with 576 rows |
| `B -> C` / `C -> B` | 7 | four matrices with 768 rows |
| `C -> C` | 7 | four matrices with 1,024 rows |
| `C -> D` / `D -> C` | 7 | four matrices with 1,536 rows |
| `D -> D` | 7 | four matrices with 2,304 rows |
| `D -> L` | 7 | `C00[768,4]`, `C01[768,6]` |
| `L -> D` | 7 | `C00[768,4]`, `C10[768,6]` |
| `A -> R` | 7 | `C00[48,4]`, `C01[48,6]` |

These are private constructors selected from the fixed topology. They are not
user-supplied layer options.

## Training Forward Contract

The selected F01-to-F01 implementation pads the four coefficient matrices to
the locked largest basis shape, stacks them, and uses one `torch.bmm` against a
single persistent packed-basis buffer. It then reshapes the four result slices
and writes them into non-overlapping quadrants of one fresh dense kernel:

```text
coefficients = stack(pad(C00), pad(C10), pad(C01), C11)
expanded = bmm(coefficients, packed_bases)
pair slice -> view(n_out,n_in,d_out,d_in,k,k)
           -> permute(n_out,d_out,n_in,d_in,k,k)
           -> reshape(d_out*n_out,d_in*n_in,k,k)
kernel = fresh dense buffer
kernel fixed quadrants = K00, K01, K10, K11
```

For a layer with `n_out*n_in` copy-pair rows, the packed coefficients are
`[4,n_out*n_in,14]`, the packed bases are `[4,14,196]`, and the valid flattened
result lengths are `[49,98,98,196]` for `00/10/01/11`; all padding outside the
corresponding coefficient/basis submatrices is exactly zero.

`bmm` has no path-planning dependency and is a CUDA autocast-to-FP16 operation
in the selected runtime. V2 measured this exact contraction and direct assembly
as the fastest predeclared path. The padding dimensions, slices, offsets, and
quadrants are fixed in `__init__`/source; there is no runtime discovery or
branching. `output = conv2d(input, kernel, bias=None, padding=k//2)` remains the
only learned convolution call.

Scalar-to-F01 lifting and F01-to-scalar projection retain their two fixed `mm`
expansions and one concatenation because the selected four-bank batching applies
only when all F0/F1 pairs are present. Missing quadrants are absent, not
zero-filled.

Every learned convolution executes exactly one dense `conv2d`. Forward has no
loop over copies or pairs, no data-dependent branch, no `.item()`, no tensor-to-
Python conversion, and no device/dtype conversion of fixed buffers. Python
helper calls only express a statically traceable tensor graph; no Python logic
survives in the compiled hot path.

The largest `D -> D` dense kernel contains 1,016,064 elements (3.88 MiB FP32 or
1.94 MiB FP16). Tests record peak temporary memory because padded coefficients,
the batched expansion, pair views/copies, and the final kernel can overlap in
lifetime.

## Buffer, Device, Dtype, And AMP Policy

- Coefficient, norm, gate, and bias parameters remain FP32 master parameters.
- Pair banks are contiguous FP32 persistent buffers. Ordinary `module.to(device)`
  moves them; training must not call `model.half()` or mutate buffer dtype.
- The EQ downsampler registers its final per-channel contiguous FP32 grouped
  weight in `__init__`; it performs no buffer `.to(...)` or channel expansion
  in forward. Autocast handles the convolution dtype.
- CUDA training uses the selected FP16 autocast policy. Autocast casts `mm` and
  `conv2d` inputs to FP16, so expanded kernels and convolution run FP16 while
  optimizer state and master parameters remain FP32.
- CPU mechanics run FP32. FP64 is confined to mathematical oracle tests.
- No `.to(device=...,dtype=...)` occurs in convolution forward.
- F0 and F1 normalization statistics are computed from unconditional
  `inputs.float()` values; outputs are cast once to the incoming dtype.
- Radial radius and sigmoid arguments are also computed in FP32 from
  `inputs.float()`, then the gate is cast once and multiplied with the original
  tensor. Use `eps=1e-4` for radius and `eps=1e-5` for normalization.
- Do not write dtype telemetry into module attributes from forward. Gate/norm
  health is collected outside the compiled hot path at bounded probe points.

This policy follows current PyTorch AMP behavior: `mm`, `matmul`, `bmm`, and
`conv2d` are CUDA FP16-autocast eligible, while reductions/norm-like operations
often require or select FP32. Relevant official references:

- https://docs.pytorch.org/docs/stable/amp.html
- https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html

## Initialization, Bias, Norm, And Gate Contracts

For an output copy and input-frequency block, initialize every coefficient
independently with

```text
std = 1 / sqrt(T_out * n_input_copies * pair_basis_dimension)
```

where `T_out` is the number of input frequencies present. This is the refreshed
Spec 0012 generalized-He rule. Hidden normalized convolutions and `mu`/`logvar`
heads use it. Hidden convolutions have no bias. `mu` and `logvar` have one scalar
bias per output F0 copy, initialized with the completed control's ordinary
uniform rule using physical `fan_in = packed_input_channels*k*k`. The final RGB
head has all coefficients and its three scalar biases exactly zero.

Field-aware normalization keeps all 40 baseline norm locations:

- F0: eight groups, mean and variance over copies in the group and spatial
  positions, one gamma and beta per copy;
- F1: four groups, no mean subtraction, RMS over grouped copies, both
  components, and spatial positions, one gamma per copy and no beta;
- initialize gamma=1 and every allowed beta=0.

Keep all 34 gate locations. F0 uses `x*sigmoid(a*x+b)`. Each F1 copy uses
`r=sqrt(u^2+v^2+1e-4)` and `sigmoid(a*r+b)*(u,v)`, with one shared `a,b` per
copy. Initialize `a=1,b=0`; use no gate gamma, no vector bias, no gate weight
decay, and the existing 0.5 gate LR multiplier.

## Residual, Resampling, RGB, And Latent Contracts

- Residual addition requires identical layout, offsets, component convention,
  spatial size, dtype, and device.
- Identity skips occur only for the same stage symbol.
- A changing stage uses the baseline branch order: fixed fieldwise resampling
  where required, selected steerable projection, field-aware norm, then add.
- Encoder main transition order remains conv -> norm -> gate -> fixed 5x5
  downsample -> conv -> norm. Skip order remains downsample -> conv -> norm.
- Decoder main transition remains bilinear x2 -> conv -> norm -> gate -> conv ->
  norm. Skip remains bilinear x2 -> conv -> norm.
- The accepted 5x5 downsample phase error is measured at primitive and transition
  levels; it is not blamed on the steerable kernel and does not trigger an
  equivariant-only 6x6 substitution.
- RGB input/output, `mu`, `logvar`, `z`, and `eps` are scalar fields. The heads
  aggregate both F0 and F1 inputs through their legal paths.
- Reparameterization and logvar clamp exactly reuse the baseline contract.

The probe implements only the convolution, field norm/gates, one identity
residual block, one encoder transition, one decoder transition, RGB lift/head,
and scalar latent heads needed to test these contracts. It does not assemble the
eight-block encoder/decoder or expose a VAE class.

## Compile And Cache Contract

The probe uses fixed batch and spatial shapes and calls
`torch.compile(..., dynamic=False)`. A single-GPU `fullgraph=True` diagnostic
must capture each probe module or fail. Inspect `TORCH_LOGS=graph_breaks,guards,recompiles`
or `tlparse`; fixed input shape, dtype, device, training mode, parameter/buffer
identity, and autocast state are expected guards. After settle, repeated calls
with the same contract must cause zero recompilations.

The real dual-T4 measurement must use the selected Spec 0011 control bundle:
channels-last activations, whole-step Inductor compilation, Python DDP reducer,
compiled autograd, compute/communication reorder, FP16 autocast with FP32 loss
and GradScaler, fused AdamW, foreach clipping, `set_to_none=True`, cuDNN
benchmark/non-deterministic kernels, TF32, high matmul precision,
gradient-as-bucket-view, 50 MiB buckets, and no buffer broadcast. Record
requested and effective readbacks. Because buffer broadcast is disabled, before
compilation or timing all-gather every persistent buffer's metadata and value
hash (or exact maximum difference) and require identical names, shapes, dtypes,
device-relative placement, and values across ranks. Stable DDP partitions are
permitted, but no model-forward graph break or post-settle recompile is
permitted. Do not add dynamic-shape support for the discarded data tail;
training remains `drop_last=True`.

Training and default evaluation always expand the current coefficients. An
optional inference-only cache is a separate explicit API:

1. after training or checkpoint load, call `materialize_eval_kernel()` under
   `no_grad`;
2. call a separate `forward_cached()` that only performs dense `conv2d`;
3. never call it while gradients or optimizer updates are active.

There is no `if self.training` cache branch in the compiled training forward and
no automatic invalidation/version counter. If the cache does not materially
help fixed-25/full validation timing, omit it.

## Focused Correctness Gate

Use the pinned ignored `escnn` checkout only in tests. For both selected
profiles and all four F01 pairs:

1. retain the existing one-copy span and sampled-kernel checks;
2. draw fixed coefficients, build the repo-owned dense block, project the same
   dense block into the escnn span, and require FP32 kernel relative residual
   `<=5e-5` and output relative RMS `<=1e-4`;
3. compare a multi-copy assembled layer for each of scalar-to-F01, F01-to-F01,
   and F01-to-scalar against an independently constructed escnn reference;
4. verify coefficient and input gradients against the reference with FP64
   relative error `<=1e-6` on a reduced one-copy CPU case;
5. verify additive bias exists only on scalar unnormalized heads and the final
   RGB head;
6. verify coefficient identities are independent across copy pairs.

Primitive and block equivariance use 15, 30, 45, 60, and 90 degrees, the same
transform convention/crop as Spec 0012, and report full-frame plus cropped
relative RMS. Kernel-only operations must satisfy the existing escnn-relative
gate. Resampling-containing transitions report their error and raw-transform
floor separately; the known 5x5 phase limitation is not a kernel failure.

## Targeted CPU Mechanics Tests

Add only focused tests for:

- exact layouts, offsets, pair-bank shapes, coefficient shapes, and parameter
  count contribution;
- padded `bmm` expansion/reshape/permutation/direct assembly against the prior
  four-`mm` test-only oracle for every F01-to-F01 fixed signature;
- packed-basis shape/content and exactly one dense `conv2d` per learned layer;
- output/input shapes for every distinct fixed signature;
- F0/F1 norm invariance/equivariance, shared-component radial gates, FP32 math,
  and finite gradients near zero radius;
- identity/transition residual compatibility and both resamplers;
- RGB lift/projection, scalar latent heads, reparameterization, and zero RGB
  initialization;
- eager FP32 forward/backward and one optimizer step;
- `torch.compile(backend="eager", fullgraph=True, dynamic=False)` mechanics on
  CPU without graph breaks.

Do not run the full repository suite unless shared infrastructure changes or a
focused failure points outside this slice. Every test docstring states the
invariant and why its failure would invalidate the experiment.

## Later Dual-T4 Benchmark And Acceptance Limits

The final remote probe requires explicit in-scope Kaggle write permission and
uses the same latest-PyPI Torch bootstrap used by the training runtime. It attaches no real
dataset and uses generated fixed-shape tensors. Test per-rank batch 4 on two
visible T4s for the high-resolution `A` identity block, `A -> B` encoder
transition, `B -> A` decoder transition, and isolated largest `D -> D`
expansion. Run matched normal-Conv2d probe blocks under the identical runtime
bundle. Exclude compile/startup from timing. For each block, prepare its eager-
EQ, compiled-EQ, and compiled-normal DDP steps; interleave one sample per mode
in window one order eager -> compiled -> normal and exact reverse order in
window two. Use 20 warmups and two untrimmed 50-sample windows, record every
sample, and require each window and its pool to satisfy the timing-CV limit.

V3 is the final transfer check of the exact selected non-equivariant runtime
flags and values with the selected singular mechanics. If it passes every
limit, accept the mechanics and stop. If it fails, stop for a new explicit
decision; do not add an arm, runtime axis, or generic tuner.

After this mechanics gate passes and the full model is separately assembled,
Spec 0011 starts from the selected probe bundle. It may use a small number of
full-model Kaggle probes to select feasible batch size and only those runtime
options shown to be architecture-sensitive. It must not blindly inherit a
baseline value that fails measurement, repeat the discarded generic search,
or treat runtime tuning as permission to change the locked SO(2) architecture.

Acceptance requires all of:

- 32 settled forward/backward optimizer updates with deterministic but rank-
  distinct inputs/targets, two visible T4s, `world_size=2`, distinct
  rank-to-device identities, finite loss/gradients/parameters, and zero AMP
  skips;
- local pre-reduction gradients must differ across ranks, while the observed
  reduced gradient and final update match an explicit two-rank mean reference;
- every persistent buffer has identical name, shape, dtype, device-relative
  placement, and value across ranks before compilation and timing;
- single-GPU model-forward `fullgraph=True` passes and dual-T4 model forward has
  zero graph breaks and zero recompiles after settle;
- compiled FP16 output relative RMS `<=5e-3` and coefficient-gradient relative
  RMS `<=2e-2` against eager FP32 on the same fixed inputs/coefficients;
- compiled pooled median step time for each composite block is no more than
  1.10x its eager-FP16 pooled median after settle;
- each EQ composite block pooled median is no more than 5.0x its matched
  compiled normal-Conv2d control;
- timed-step coefficient of variation `<=10%` per rank;
- peak reserved VRAM `<14.5 GiB` and peak allocated VRAM `<13.5 GiB` per T4 at
  batch 4, leaving operational headroom on a 16 GiB device;
- no hidden host synchronization in the timed body and no runtime basis/layout
  construction.

The isolated D-to-D expansion/complete-forward fraction remains recorded with
raw samples as diagnostic evidence, but is not a pass/fail condition. A large
internal fraction is actionable only if it causes a representative composite
or later full-model operational failure.

Passing these limits accepts the concrete convolution mechanics and permits a
separately authorized full model to be coded. It does **not** establish full-topology activation
memory, optimizer/DDP-bucket VRAM, end-to-end step time, or epoch time. Those
must be measured on the assembled model by Spec 0011 before any run. The
isolated `<14.5/<13.5 GiB` limits are only fail-fast bounds for an obviously
unusable block implementation.

If any final correctness, performance, or memory limit fails, stop and report
the evidence. Do not revise mechanics, runtime, field multiplicities, or radial
profiles within this spec. Spec 0011 and full-model coding remain blocked until
a new explicit decision or a passing Spec 0013 probe.

## Outputs And Acceptance Artifacts

The local implementation produces:

- the fixed probe modules under `src/eqvae/models/`, not exported as a generic
  public API;
- focused CPU tests;
- one compact tracked CPU mechanics/equivariance summary;
- one narrow guarded singular four-path runner; no arm comparison, extensible
  controller, or generic configuration system.

## Acceptance Criteria Before Full-VAE Coding

1. Pass: all offline/layout construction is absent from training forward.
2. Pass: F01-to-F01 convolutions use the selected padded `bmm` plus
   direct assembly; scalar boundary maps retain fixed `mm`; every learned layer
   uses exactly one dense `conv2d`.
3. Pass: focused escnn, gradient, layout, norm, gate, bias, residual,
   resampling, RGB, and scalar-latent checks pass.
4. Fail: the singular dual-T4 final probe passes all correctness and operational
   ratios but fails the unchanged 10% timing-CV gate.
5. Pass: counts remain exactly 1,172,304 coefficients, 3,600 norm parameters,
   4,096 gate parameters, 35 scalar biases, and 1,180,035 total learned
   parameters for the eventual 43-convolution topology.
6. Pass: v3 evidence and handoffs are updated; do not assemble the full VAE.

Only then may a separate implementation step assemble the full equivariant VAE.

## Verification Commands

The exact local verification commands are:

```bash
.venv/bin/pytest -q tests/test_so2_basis.py tests/test_so2_architecture_probe.py tests/test_so2_architecture_probe_kernel.py
.venv/bin/ruff check src/eqvae/models/so2_architecture_probe.py src/eqvae/benchmarking/so2_architecture_probe.py tests/test_so2_architecture_probe.py tests/test_so2_architecture_probe_kernel.py kaggle/kernels/so2_architecture_probe/run_template.py
.venv/bin/ruff format --check src/eqvae/models/so2_architecture_probe.py src/eqvae/benchmarking/so2_architecture_probe.py tests/test_so2_architecture_probe.py tests/test_so2_architecture_probe_kernel.py kaggle/kernels/so2_architecture_probe/run_template.py
.venv/bin/basedpyright src/eqvae/models/so2_architecture_probe.py src/eqvae/benchmarking/so2_architecture_probe.py tests/test_so2_architecture_probe.py tests/test_so2_architecture_probe_kernel.py kaggle/kernels/so2_architecture_probe/run_template.py
.venv/bin/python scripts/select_so2_basis.py --refresh-layout --check
./scripts/kaggle_kernel.sh preflight-so2-architecture-probe
./scripts/agent_preflight.sh
git diff --check
```

The v3 focused suite passes `69 passed` with 329 pinned-escnn/SciPy deprecation
warnings. Source hashes, Ruff format/lint, BasedPyright, the exact basis check,
artifact check, local Kaggle preflight, agent preflight, and `git diff --check`
pass. Kaggle kernel v3 ran from clean commit `c823a7e`; its source-bound summary
and raw artifact/log hashes are tracked in the v3 evidence file above.

## Implementation Blockers

The final probe failed only the timing-CV gate. Progress requires a new explicit
decision about whether the cross-rank correlated noise invalidates this CV
protocol; no rerun, arm, runtime-axis change, tolerance change, or full-VAE work
is authorized meanwhile.

## Adversarial Review Findings

Fresh read-only mathematical review found no basis/layout/norm/gate premise
error. It required all named coefficient gradients, all pair/angle
escnn-relative gates, instantiated generalized-He/bias/count checks, and source
fingerprints; each was added. Fresh compile/performance/scope review required
exact selected-runtime JSON provenance plus live compiler/DDP readback,
batch-4 fullgraph coverage and initial-break recording, load-bearing eager and
control CV plus AMP/finiteness rows, and a fresh embedded runner; each was
added. The reviewer also caught an early raw-module timing draft; the final
probe times DDP-wrapped modules under the selected runtime and measures
assembly in the compiled FP16 path.

Fresh v3 mathematical review found no defect in packed-basis ordering, padded
`bmm`, direct quadrants, fixed signatures, FP64 autograd comparison,
initialization/count/buffer policy, or the tolerance-based contract. Fresh v3
performance/scope review required and verified compiled-DDP cross-rank parameter
agreement, max-reduced initial graph breaks, exact rank/device and raw-sample
schemas, and two actual Tesla T4s. No substantive finding remains.

After v1, fresh read-only mathematical review identified the unscaled FP16
gradient diagnostic and required GradScaler, compiled autograd, unscaled master
gradients, and named worst-gradient evidence. Fresh performance/scope review
identified host synchronizations in timed telemetry and required the fixed
three-arm, two-window comparison with raw samples and unchanged limits. The
final reviews also made selection fail closed on global accuracy failures,
bound buffer/graph/VRAM evidence to the candidates, made each window's assembly
fraction load-bearing, pinned the reviewed Spec 0011 runtime SHA-256, and
versioned the v2 download destination. The local implementation and tests
include each correction; neither review found grounds to change the
architecture or tolerance.

## Known Risks

- Dense packed widths make convolution compute much larger than parameter count
  suggests; the probe may be correct but too slow.
- Autocast or compiler changes in a newer Torch release may alter contraction
  lowering; the benchmark records and uses the exact upgraded runtime.
- Packed expansion and direct-assembly temporaries may matter more than analytic
  expansion MACs.
- The retained 5x5 downsample has known sampled-grid phase error.
- A cache used before optimizer completion would silently serve stale kernels;
  the default path therefore never caches.

## Adversarial Checks

- Swap F1 component order or a kernel-coordinate sign and verify escnn tests fail.
- Transpose copy/irrep axes before reshape and verify the multi-copy oracle fails.
- Tie two copy-pair coefficient rows and verify the independence test fails.
- Permit F1 bias, componentwise activation, or mean subtraction and verify
  equivariance tests fail.
- Move `.float()` after squaring in radial norm and verify near-limit FP16 data
  exposes the overflow risk.
- Change batch, spatial size, autocast state, or train/eval path and confirm the
  compile guard/recompile diagnostic reports it instead of silently broadening
  the supported surface.
- Mutate the 5x5 downsampler and ensure its error is attributed separately from
  the convolution basis.

## Open Questions

Whether to retain the per-window 10% CV protocol after its strongly correlated
two-rank failure. Any revision or rerun requires a new explicit user/spec
decision. Full-VAE coding remains separately unauthorized.

## Related Files

- `docs/specs/0012-continuous-so2-vae-architecture.md`
- `docs/decisions/0004-so2-gaussian-ring-kernel-basis.md`
- `docs/equivariant_vae_transition_plan.md`
- `configs/spec0012/so2_basis_manifest.json`
- `docs/data/spec0012_so2_basis_audit.json`
- `docs/data/spec0013_so2_dual_t4_probe_v1.json`
- `docs/data/spec0013_so2_dual_t4_probe_v2.json`
- `src/eqvae/models/so2_basis.py`
- `src/eqvae/models/non_equivariant_vae.py`
- `src/eqvae/models/resampling.py`
