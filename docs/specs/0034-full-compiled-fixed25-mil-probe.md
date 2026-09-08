# Spec 0034: Full Compiled Fixed-25 MIL Probe

Status: version 7 completed; correctness and full-bag capacity accepted
Implementation readiness: architecture/runtime capacity gate passed; learning remains separate
Owner/workstream: Spec 0033 full-network integration
Last updated: 2026-09-03

Version-7 retry authority: `spec0034_pinned_torch_retry_v7_authorized` was
granted by the user and consumed on 2026-09-03 by private Kaggle kernel
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/7` using the exact
locally verified PyTorch 2.14.0/cu130 repair. Its immutable launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0007.json`
(SHA-256 `e201f902cc135f897b50a1fa5e7575d22eb912830bd0c880ec088242afded530`).
The launch-time spec SHA-256 is
`2a0e6c66ab6c16e176b3986f5f7426dd099aee979b1d586c885e15f8a5d433c5`.
The immediate authenticated status was `RUNNING`; the later explicitly checked
terminal status was `COMPLETE`. This authority does not permit learning,
sealed-test access, architecture changes, or another retry.

## Version 7 Outcome

Version 7 passed the probe's own correctness and capacity contract under exact
`torch 2.14.0+cu130` / CUDA 13.0. Authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v7/`. Result, log and
download-receipt SHA-256 values are respectively
`26a783b053aaf503fb8ba28c03e804c906d1e4903d6b3b54493f3230d22e1551`,
`29d0fbb93f878db90095175409be660b7becc3bf617aa33203a3eb2270be14ae`
and `1d6646e2a3a9615ed9f4228bfdb402b0b4a12a6cef96c546d09e94469a7e0cf3`.

The small correctness gate passed with one compiled numerical graph and zero
graph breaks. Each real branch then loaded all 32,595 patches and the identical
793,081-edge radius-two graph, used exactly two numerical graphs including the
64-patch dynamic-reuse specialization, and reported zero graph breaks. Both
branches had one matching standard-GradScaler warmup skip at scale 32768,
recovered to 16384, and committed all five measured steps with finite losses.

The normal-latent branch measured 568.25 ms mean wall time per full training
step, 5,724,899,328 peak allocated bytes, 6,811,549,696 peak reserved bytes and
8,662,089,728 bytes conservative headroom. The SO(2)-latent branch measured
475.11 ms, 5,724,899,328 peak allocated, 6,769,606,656 peak reserved and
8,739,684,352 bytes conservative headroom. Both exceeded the 512 MiB headroom
gate. `accepted_capacity=true`. This establishes capacity and internally gated
compiled behavior for the fixed probe workload; it is not classifier learning,
generalization, sealed-test evaluation, or a matched end-to-end speedup claim.

Version-6 retry authority: `spec0034_recompile_limit3_retry_v6_authorized` was
granted by the user and consumed on 2026-09-03 by private Kaggle kernel
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/6`. Its immutable
launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0006.json`
(SHA-256 `daff34fe7ebe183ed4217c683ab6f42d59e263221a64d61ba103f841ba200ff1`).
The launch-time spec SHA-256 is
`6351e833cfd61efa03b80b7d10bb2aa3aa628b49952ec5f5de2a2b21cb776efc`.
It changes only PyTorch's `recompile_limit` from 1 to 3; runtime acceptance
still permits at most two unique numerical graphs and rejects a third. It does
not authorize learning, sealed-test access or any further retry.

## Version 6 Outcome

Version 6 completed `ERROR` during `upgrade_torch`, before importing the model
or exercising the recompile-limit change. Authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v6/`. Result, log and
download-receipt SHA-256 values are respectively
`c0a31a59f8f6a466012e617a39a0437089a72566355cef0ee6755962744695e6`,
`fc41cb458f38b8ffe9a2a2ea5be4b33b250c08f83ea63ba07b890c17b2e690df`
and `4ee2be6ba8e94664da1216b90ab0d973966f477bd2c8a114679703ae6b4e93bf`.

Kaggle began with preinstalled `torch 2.10.0+cu128`. The runner's unpinned
`pip install --upgrade torch torchvision torchaudio` selected `torch 2.14.0`
and a floating CUDA-13 dependency set. After downloading the packages, pip
rejected one package because its received SHA-256 differed from the index's
expected SHA-256, then returned exit status 1. The artifact cannot identify the
specific package. Pip correctly refused installation; this run provides no
model, AMP, compilation, memory or capacity evidence. Requiring a floating
latest-stack network installation in every probe is a reproducibility defect.
The local repair pins only `torch==2.14.0` from PyTorch's official cu130 wheel
index, disables pip's cache for the install, and verifies imported torch 2.14.0
plus CUDA 13.0 before activating the model. It does not reinstall unused
`torchvision` or `torchaudio`. Version 2.14.0 is the latest release already
authenticated end to end by version 5; this removes floating dependency
selection but is not new remote evidence.

Version-5 retry authority: `spec0034_grad_scaler_retry_v5_authorized` was
granted by the user and consumed on 2026-09-03 by private Kaggle kernel
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/5` using the
locally verified standard-GradScaler repair. Its immutable launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0005.json`
(SHA-256 `67b0e09538f2371a9a386a06d03a1033a810e2340b2cb16db6d01fed48f09a49`).
The launch-time spec SHA-256 is
`0f2cdb794f33b787963694c20af0dc0194e77da59927c42f9fc28c818c008996`.
It does not authorize learning, sealed-test access or any further retry.

Version-4 retry authority: `spec0034_training_effect_retry_v4_authorized` was
granted by the user and consumed on 2026-09-03 by private Kaggle kernel
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/4`. Its immutable
launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0004.json`
(SHA-256 `ac8b16beccd4330f21b1d829799a1db55a0195d3164196b5814d5129a9a20f00`).
The launch-time spec SHA-256 is
`471cbc19e1d97b0f857aa75f0e62d0c4e6138d4bb928443db6971e5654367d11`.
The authority covered one private capacity/optimization probe only; it did not
authorize learning or sealed-test access. The earlier
`spec0034_permissive_dynamic_retry_v3_authorized` authority is consumed.
Kaggle accepted private
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/3` on 2026-09-03
using the locally verified permissive dynamic fix; it later completed `ERROR`.
Its immutable launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0003.json`.
The launch-time spec SHA-256 is
`626ea3b3a8016944639675d3a67ac1d081f503d0218bdd02dc0a06e650e9b7cbf`.
Beyond the one version-6 attempt above, no repeat, learning campaign,
sealed-test access, or production-backend change is authorized.

## Version 5 Outcome

Kaggle reported `ERROR`; the probe artifact records controlled status
`rejected`. Authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v5/`. Result, log and download-receipt
SHA-256 values are respectively
`8eb645a19c17b830534b0cee546f26598e88a07f5f1aeff6e81ad22a5c25c16b`,
`f3cccdb76e231aae9133890979fd1c93f6808bdf2740957da82ca23e0ec1ba5d`
and `95eb09c5d76ed14c82eec18dba25c21db6d7196cb0909444257fca1c2f9d9037`.

The five-attempt small-model correctness gate passed with one compiled graph
and zero graph breaks. At attempt 1, all three AMP arms identically found
infinities in `local_blocks.0.ffn.output.weight`, skipped AdamW and reduced the
scale from 32768 to 16384. All four later AMP attempts committed, matching the
intended standard GradScaler behavior and skip-history gate.

Both exact 32,595-patch latent bags and the 793,081-edge graph loaded. On each
T4, the real-size compiled numerical callable completed finite forward/backward
with optimizer state resident. Each branch then failed on the deliberate
64-patch dynamic-reuse call with `FailOnRecompileLimitHit`. The wrapper supplied
`recompile_limit=1`, which rejected the first required second specialization,
while this spec explicitly permits one small-N plus one large-N graph. This is
an internal probe-contract defect, not AMP, model, attention, OOM or graph-break
evidence. No full-bag optimizer step, settled timing, peak-memory measurement or
headroom result was produced. A future local repair must allow the two declared
specializations without weakening the third-specialization failure; no remote
retry is authorized.

## Version 4 Outcome

Version 4 completed `ERROR` in `small_full_model_correctness`. Authenticated
evidence is under `runs/kaggle/spec0034_full_compile_fixed25_v4/`. The result,
log and download-receipt SHA-256 values are respectively
`b9fcd38c1e46ca406e16e9d4ed55930698f688a7b229e6a667f7bcee86d12722`,
`b40aa4c7f0d98fa5e7a9bff3e3386a1730b2b49a4f7118c34d4fe5114bc2c409`
and `84eb3d82151c6789de019a58322263f8a8f96597e3f12d11ff3ce1d7d75d3ff2`.
No full bag was loaded and no capacity, memory or compiled-backend result was
produced.

The exact failure is a nonfinite parameter gradient before unscale and before
the failing arm's AdamW step. Both FP32 and AMP paths multiplied loss by the
fixed initial scale `32768`, checked the still-scaled gradients for finiteness,
and aborted instead of applying ordinary GradScaler skip/backoff semantics.
Timing and lazy compilation strongly indicate step-1 eager AMP, but the artifact
cannot prove the arm, logical step or parameter because the exception collapsed
all gradients into one boolean. Do not attribute this result to Inductor.

The local repair moves scaling outside the compiled loss callable and uses the
supported `torch.amp.GradScaler` sequence. An overflowing attempt skips its
optimizer update, backs off the scale and advances; it is not retried or
requeued. Eager AMP, replay AMP and compiled AMP must have identical skip/scale
histories. The gate persists arm, logical attempt, scale and every offending
gradient; committed-step counters advance only when AdamW runs. Selecting a
lower fixed scale after this failure would be unsupported. Only the version-5
retry authorized above may test this repair on Kaggle.

## Version 3 Outcome

Authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v3/`. The artifact SHA-256 is
`7ae1c1578b78bf9f45f007e158764ff1bc344ddba04c6e0a5043e1639c56832a`;
the downloaded log SHA-256 is
`b40ac9ffe92291fb308de5cce90a6e07ee90f414ed953a6a7cda87c3178fd2d4`.
The permissive dynamic repair worked: the compiled 25-patch end-to-end
forward/backward produced one unique graph and zero graph breaks, with no
constraint violation. Loss, logits and latent gradients passed. The run stopped
at `small_full_model_correctness` before loading the real 32,595-patch bag, so it
is not capacity, memory or optimizer evidence.

The exact failure was the predeclared gradient relative-L2 limit of `0.002`
against the pure-FP32 reference. Five AMP rows narrowly exceeded it:

| Path | Relative L2 | Cosine | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| eager CLS query weight | 0.002298 | 0.999997 | 0.000120 |
| compiled local-0 relative bias | 0.002222 | 0.999998 | 0.000072 |
| compiled local-0 null key | 0.002425 | 0.999997 | 0.000085 |
| compiled local-0 null bias | 0.002011 | 0.999998 | 0.000149 |
| compiled CLS query weight | 0.002016 | 0.999999 | 0.000099 |

The eager AMP path itself failing one row rules out a compiler-only numerical
divergence. Two global-token gradient rows failed the separately reported
elementwise `allclose` field but passed the actual gradient gate through their
relative-L2 and cosine values; they did not cause termination. Do not relabel
this as a passing correctness result, but do not attribute it to the repaired
dynamic annotation either.

The earlier `spec0034_standard_norm_retry_v2_authorized` authority is consumed by
`maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/2`. Its immutable
receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/v0002.json`.
The launch-time spec SHA-256 embedded in version 2 is
`1374642e6305b4875b1aaf60fde90c7508192ae94db907ff47ee47a3f2108842`.

## Version 2 Outcome

Version 2 completed `ERROR`; authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v2/`. Standard PyTorch normalization
passed the eager CUDA path. The run then failed while compiling the 25-patch
correctness case, before real-bag loading or capacity measurement, because the
runner declared its N axis dynamic over `1..32595`, while tracing also emitted
the guard `192*N < 5242880`. Those constraints are incompatible: the latter is
false for part of the declared range (strictly, it permits integer
`N <= 27306`). The actual correctness input remained `N=25`; neither the actual
size nor the declared maximum changed. Dynamo raised `ConstraintViolationError`
because it could not validate the traced graph for every size that
`mark_dynamic` asserted was admissible.

PyTorch 2.14 source identifies the exact threshold: Inductor's optional
`MixOrderReduction.can_fuse` profitability check uses `5 * 2**20` elements and
guards whether `nrow*ncol` is above it. For an `[N,192]` LayerNorm-backward
reduction, tracing the small case selected the below-threshold/no-fusion branch
and recorded its negation as `192*N < 5242880`. PyTorch describes this as a
non-critical L2-cache performance check, not a capacity condition. AOTAutograd
may compile the backward graph ahead of the explicit `loss.backward()` call so
that backward-generated guards are attached to the forward cache entry. The
existing non-verbose log cannot identify which of the model's `[N,192]`
LayerNorm instances triggered the check.

The expression `192*N` is the element count of a two-dimensional `[N,192]`
matrix, not a tensor axis or an expansion. The implementation never reshapes a
patch tensor to `[192*N]`: CNN flattening maps `[N,192,1,1]` to `[N,192]`, and
attention maps `[N,192]` to `[N,6,32]`. The four marked input axes are latent
axis zero plus axis zero of `neighbor_index`, `neighbor_valid`, and
`radial_code`; model validation and local-attention broadcasting require those
four axes to be equal. All other input axes are static. Five LayerNorm sites
operate on dynamic `[N,192]` tensors: both norms in each local block and the
global summary's patch norm.

This is a dynamic-annotation failure in the probe, not a model, normalization,
attention, OOM or capacity result. It does not establish whether a separate
real-size graph at `N=32595` would compile. Merely tracing at the real size with
the same `1..32595` annotation would select the opposite guard and remain
incompatible with the full range. The selected local fix replaces strict
`mark_dynamic(..., min=1, max=32595)` assertions with axis-zero
`maybe_mark_dynamic` hints. This preserves `N` as the only dynamic dimension
while allowing Dynamo/Inductor to cache a second specialization when an
internal scheduling guard cannot span both sides of the 27,306/27,307 boundary.
The same compiled callable may therefore own one or two numerical graphs; graph
breaks and a third specialization remain failures. No retry is authorized here.

## Version 1 Outcome

Private `maximshtefan/eqvae-wsi45630-full-compiled-fixed25-mil-probe/1`
completed `ERROR`. Authenticated evidence is under
`runs/kaggle/spec0034_full_compile_fixed25_v1/`. The run failed during the
small full-model correctness phase, before real-bag loading, compilation,
measurement or optimizer execution, with `RuntimeError: expected scalar type
Half but found Float` in the first `LowPrecisionGroupNorm`.

This is an implementation defect, not capacity evidence. CUDA autocast supplied
an FP16 convolution output, but the wrapper disabled autocast and passed it to
native GroupNorm with FP32 affine tensors. PyTorch 2.14 CUDA rejects that mixed
dtype combination. `LowPrecisionLayerNorm` has the same latent defect. CPU
PyTorch accepted the combination, so the existing CPU-focused tests could not
establish the required T4 behavior. Version 2 removes both wrappers and uses
ordinary PyTorch GroupNorm/LayerNorm under AMP; this is its only change from the
version-1 model path.

## Purpose

Test whether Spec 0033's correct and fast whole-bag fixed-25 Inductor attention
works inside the complete width-192 MIL training path on the exact 32,595-patch
WSI45630 bag. Measure end-to-end compilation, optimizer compatibility, memory
headroom and settled step time before selecting it for production.

## Fixed Architecture

- Frozen `[N,16,32,32]` posterior means transferred to each T4 as FP16 in
  channels-last format.
- Current `LocalGlobalMILClassifier`: convolution/standard AMP normalization,
  two local Transformer blocks, 16 REG plus CLS, calibrated sigmoid global
  summary, CLS-only softmax read, FP32 final norm/logits and five-class loss.
- Both local modules are replaced state-compatibly by
  `WholeBagFixed25Attention`; no architecture or parameter-count change.
- Exact physical radius-two `[N,25]` graph, learned radial score bias and one
  zero-value null per query/head remain unchanged.

## Compilation And Optimization Contract

1. Compile model plus FP32 cross-entropy as one `fullgraph=True`, selectively
   dynamic Inductor region under `max-autotune-no-cudagraphs`. Keep standard
   GradScaler orchestration outside the compiled numerical callable.
2. Give permissive dynamic hints only to dimension zero of latents and the three
   graph tensors; all channel, head, degree and class axes stay static.
3. Let AOTAutograd generate the backward graph. Permit at most two cached
   numerical specializations across the real and small bags, require zero graph
   breaks, and reject a third specialization or eager fallback.
4. Use `zero_grad(set_to_none=True)`, FP32 master weights, CUDA AMP FP16,
   channels-last convolution, cuDNN benchmark, nondeterministic fast kernels,
   expandable allocator segments and the current latest PyTorch CUDA stack.
   For this probe, that stack is now exact: install only `torch==2.14.0` from
   `https://download.pytorch.org/whl/cu130`, then require torch 2.14.0 and CUDA
   13.0 at runtime. Do not request an unpinned upgrade or install unused domain
   libraries.
5. Use semantic AdamW groups and resident optimizer state with native
   `fused=True` AdamW. Compile only model/loss/backward; do not add a second,
   separately confounded compiled-optimizer experiment to this capacity probe.
6. CUDA graphs stay disabled because bag length is dynamic. No activation
   checkpointing is used in the primary row; OOM is evidence, not permission to
   silently alter the computation.

## Correctness And Capacity Gate

- On a hostile small physical graph, run five matched training attempts for four
  byte-identically initialized fixed-25 arms: eager FP32, eager AMP, repeated
  eager AMP, and compiled AMP. Vary the synthetic input and label across steps
  so Adam moments and gradient directions evolve.
- AMP arms use independent standard GradScalers with the same initial state,
  fixed-25 backend, parameter groups and native fused AdamW; FP32 uses a disabled
  scaler. FP32 and repeat arms are measured controls rather than oracles; no
  per-parameter decimal tolerance gates the run.
- The three AMP arms must have identical `(skipped, scale-before, scale-after)`
  histories. A skipped attempt advances the input sequence but not AdamW's
  committed-step counter; it is neither retried nor requeued. Missing gradients
  remain an exact failure, while every nonfinite gradient is recorded before
  GradScaler performs its normal skip/backoff.
- After every step, gate actual updates, cumulative parameter trajectories and
  first/second Adam moments separately for every named parameter. Record unscaled
  gradient differences diagnostically, without letting gradients that AdamW
  neutralizes veto an equivalent training update. Evaluate every state through
  the common eager-FP32 fixed-25 path on the whole five-case panel. Gate
  probabilities and losses separately; record argmax predictions diagnostically
  so an otherwise acceptable near-tie class flip is not an arbitrary veto.
- For every parameter/state metric at every step, compiler-induced training
  effect is `||compiled_amp-eager_amp||`; the accepted effect is the larger of
  measured `||eager_amp-fp32||` and
  `||eager_amp_repeat-eager_amp||`. This directly tests the change introduced by
  compilation against observed precision/repeat effects; an equal-radius move
  opposite to eager AMP therefore exceeds the envelope. If the measured envelope
  is exactly zero, require exact equality.
  Missing/nonfinite state, unequal step counters, a parameter never updated in
  an arm, graph breaks, or extra compiled graphs fail exactly.
- Run both normal- and SO(2)-latent branches independently, one per T4, with
  separate models, numerical callables and optimizers. They never synchronize
  gradients or gate one another. Capture a branch exception as a failed row and
  continue the other branch; overall acceptance still requires both rows.
- Materialize AdamW state before resetting peak memory. Execute two unmeasured
  full-bag attempts, a different-size dynamic-reuse compilation check, then five
  measured full-bag attempts per branch. This capacity benchmark intentionally
  repeats its one fixed WSI workload and is not a training-epoch iterator or an
  exception to Spec 0032's no-requeue learning policy. Report every skip and
  scale transition; all five measured attempts must commit for timing/capacity
  acceptance.
- Record compile time, forward/backward/update timing, losses, finite-gradient
  coverage, unique graphs/recompiles/graph breaks, generated kernel count,
  peak allocated/reserved memory and unallocated headroom.
- Acceptance requires both full-bag branches to finish with finite losses and
  gradients, at least 512 MiB unallocated headroom, at most two cached numerical
  specializations, zero graph breaks, and no fallback in the numerical region.

## Inputs And Outputs

- Input dataset: `maximusshtefan/eqvae-wsi45630-capacity-inputs/1`.
- Latent kernel sources remain exact owner-qualified version-1 locators from
  that input contract; the authenticated Kaggle actor owns only the new kernel.
- Output: `spec0034_full_compiled_fixed25_mil_probe.json`, plus the immutable
  account-portable launch/download receipts.

## Verification Policy

This is an isolated disposable Kaggle probe. Its launch gate is touched-file
Ruff/BasedPyright, focused builder/package/model tests, `bash -n` for launcher
changes, kernel validation, `git diff --check`, and independent adversarial
review when agent capacity is available. Do not run the repository-wide suite
solely for this probe.

The permissive-dynamic repair passed its focused checks before version 3. The
standard-GradScaler/runtime repair and replacement training-effect gate pass 25 focused
probe/precision tests, touched-file Ruff and BasedPyright, representative
training-state mutation checks, and package rebuild/validation. The explicit
version-4, version-5 and version-6 authorities above are consumed.

Three clean-context reviews rejected raising the old tolerance. Their accepted
estimand is the control-relative optimizer trajectory above: same backend across
arms, real loss scaling and AdamW, per-parameter/per-step accounting, repeated AMP
control, and functional post-update evaluation. They also required removing the
dummy zero-gradient compiled-optimizer probe. A small-N pass still cannot prove
the large-N specialization numerically; without a real-size differential, a
future successful run may establish capacity and absolute step cost but not
production equivalence or an end-to-end backend speedup ratio.
