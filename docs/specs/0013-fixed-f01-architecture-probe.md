# Spec 0013: Fixed F01 Architecture Probe

Status: locked / implementation-ready
Owner/workstream: one-off continuous-`SO(2)` architecture probe
Last updated: 2026-08-12

## Purpose

Prove one compile-friendly implementation for the selected equal-copy F01
architecture before the full equivariant VAE is coded. This is a gate for one
experiment, not a reusable equivariance library.

Spec 0012 already selected `9-low` for the stem, `7-low` everywhere else, and
the fixed hidden copy pairs `(16,16)`, `(24,24)`, `(32,32)`, `(48,48)`. This
spec does not reopen that result. It fixes how those coefficients become one
dense kernel, how fixed fields are packed, and how the surrounding operations
behave under AMP and `torch.compile`.

The count refresh exposes an important cost premise: `1,180,035` learned
parameters are only 29.81% of the baseline cap, but the packed-width 9x9/7x7
dense convolutions cost `159,837,585,408` MAC/sample, 4.383x the baseline's
learned-convolution MACs. Parameter compliance is not compute matching. The
probe must measure this implementation honestly; it must not change the locked
layout to hide the cost.

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

Use `torch.mm`, not a multi-input `einsum`, for every pair expansion:

```text
flat_block = coefficients @ pair_basis
flat_block -> view(n_out,n_in,d_out,d_in,k,k)
           -> permute(n_out,d_out,n_in,d_in,k,k)
           -> reshape(d_out*n_out,d_in*n_in,k,k)
```

`mm` makes the contraction explicit, has no path-planning dependency, and is a
CUDA autocast-to-FP16 operation in current PyTorch. Each F01-to-F01 layer uses
exactly four `mm` calls. Assemble the quadrants with three concatenations:

```text
top    = cat((K00, K01), dim=input_channel)
bottom = cat((K10, K11), dim=input_channel)
kernel = cat((top, bottom), dim=output_channel)
output = conv2d(input, kernel, bias=None, padding=k//2)
```

Scalar-to-F01 lifting uses one output-channel concatenation. F01-to-scalar
projection uses one input-channel concatenation. Missing quadrants are absent,
not zero-filled. No output-sized preallocation followed by slice assignment is
allowed unless the dual-T4 probe proves it materially faster and the spec is
updated first; such writes create another large temporary and complicate
autograd/functionalization.

Every learned convolution executes exactly one dense `conv2d`. Forward has no
loop over copies or pairs, no data-dependent branch, no `.item()`, no tensor-to-
Python conversion, and no device/dtype conversion of fixed buffers. A private
helper may remove source duplication, but the four pair calls and assembly are
statically unrolled for each of the three map kinds.

The largest `D -> D` dense kernel contains 1,016,064 elements (3.88 MiB FP32 or
1.94 MiB FP16). Tests record peak temporary memory because concatenation can
briefly retain pair blocks, row blocks, and the final kernel together.

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
- `mm` expansion/reshape/permutation against a direct loop oracle in tests;
- static three-cat assembly and exactly one dense `conv2d` per learned layer;
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

The later remote probe requires fresh Kaggle write permission and the same
latest-PyPI Torch bootstrap used by the training runtime. It attaches no real
dataset and uses generated fixed-shape tensors. Test per-rank batch 4 on two
visible T4s for the high-resolution `A` identity block, `A -> B` encoder
transition, `B -> A` decoder transition, and isolated largest `D -> D`
expansion. Run matched normal-Conv2d probe blocks under the identical runtime
bundle. Exclude compile/startup from timing.

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
- compiled median step time for each composite block is no more than 1.10x its
  eager-FP16 median after settle; coefficient expansion plus assembly is no more
  than 10% of isolated `D -> D` expanded-kernel-plus-convolution forward time;
- each EQ composite block median and the topology-position-weighted sum of the
  measured signatures are no more than 5.0x their matched compiled normal-
  Conv2d controls; this is a mechanics projection, not an epoch-time claim;
- timed-step coefficient of variation `<=10%` per rank;
- peak reserved VRAM `<14.5 GiB` and peak allocated VRAM `<13.5 GiB` per T4 at
  batch 4, leaving operational headroom on a 16 GiB device;
- no hidden host synchronization in the timed body and no runtime basis/layout
  construction.

Passing these limits selects the concrete convolution mechanics and permits the
full model to be coded. It does **not** establish full-topology activation
memory, optimizer/DDP-bucket VRAM, end-to-end step time, or epoch time. Those
must be measured on the assembled model by Spec 0011 before any run. The
isolated `<14.5/<13.5 GiB` limits are only fail-fast bounds for an obviously
unusable block implementation.

If correctness fails, fix the implementation or this spec's mathematical
premise before full-VAE work. If correctness passes but a performance/memory
limit fails, measure only the concrete `mm`/assembly issue responsible and
revise the contraction or assembly inside this probe; do not change field
multiplicities, radial profiles, or build a generic tuning framework. Spec 0011
and full-model coding remain blocked until the revised Spec 0013 probe passes
every limit.

## Outputs And Acceptance Artifacts

Later implementation produces:

- the fixed probe modules under `src/eqvae/models/`, not exported as a generic
  public API;
- focused CPU tests;
- one compact tracked CPU mechanics/equivariance summary;
- one compact tracked dual-T4 summary with runtime fingerprint, graph counters,
  accuracy, latency, temporary-memory, VRAM, and pass/fail fields;
- no retained one-use benchmark controller or generic configuration system.

## Acceptance Criteria Before Full-VAE Coding

1. All offline/layout construction is absent from training forward.
2. Every convolution uses the fixed `mm` contractions, minimal static assembly,
   and exactly one dense `conv2d`.
3. Focused escnn, gradient, layout, norm, gate, bias, residual, resampling, RGB,
   and scalar-latent checks pass.
4. CPU fullgraph mechanics and the explicit dual-T4 limits pass.
5. Counts remain exactly 1,172,304 coefficients, 3,600 norm parameters, 4,096
   gate parameters, 35 scalar biases, and 1,180,035 total learned parameters for
   the eventual 43-convolution topology.
6. Spec 0012, this spec, the compact evidence, and `CURRENT.md` are updated with
   the result and exact next full-VAE step.

Only then may a separate implementation step assemble the full equivariant VAE.

## Verification Commands

The implementation slice will define the exact focused filenames. Its minimum
local commands are:

```bash
.venv/bin/python -m pytest -q tests/test_so2_basis.py tests/test_so2_architecture_probe.py
.venv/bin/ruff check <touched Python files>
.venv/bin/ruff format --check <touched Python files>
./scripts/agent_preflight.sh
git diff --check
```

The one-use Kaggle probe command is added only when its guarded kernel exists;
remote writes remain explicitly permission-gated.

## Implementation Blockers

None for the local probe. The dual-T4 portion requires explicit Kaggle remote-
write permission when the probe implementation is ready.

## Known Risks

- Dense packed widths make convolution compute much larger than parameter count
  suggests; the probe may be correct but too slow.
- Autocast or compiler changes in a newer Torch release may alter contraction
  lowering; the benchmark records and uses the exact upgraded runtime.
- Concatenation temporaries may matter more than analytic expansion MACs.
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

None that block implementation. The dual-T4 result may force a narrow revision
of the contraction/assembly choice, but not a new architecture search.

## Related Files

- `docs/specs/0012-continuous-so2-vae-architecture.md`
- `docs/decisions/0004-so2-gaussian-ring-kernel-basis.md`
- `docs/equivariant_vae_transition_plan.md`
- `configs/spec0012/so2_basis_manifest.json`
- `docs/data/spec0012_so2_basis_audit.json`
- `src/eqvae/models/so2_basis.py`
- `src/eqvae/models/non_equivariant_vae.py`
- `src/eqvae/models/resampling.py`
