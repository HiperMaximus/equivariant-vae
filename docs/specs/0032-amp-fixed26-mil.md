# Spec 0032: AMP-First Fixed-26 Local-Global MIL

Status: implemented / locally accepted
Implementation readiness: local model complete; CUDA capacity execution is not authorized
Owner/workstream: Spec 0026 post-OOM memory/runtime repair
Last updated: 2026-09-04

## Purpose

Repair the exact Spec 0026 architecture after Spec 0030 OOMed in local block 2.
Keep its topology and attention algebra, but make precision and kernel choices
match the real Tesla T4: FP16 resident latents, fused fixed-degree local SDPA,
FP32 master parameters, and standard PyTorch AMP policy.

## Evidence And Selected Backend

Spec 0028 already measured the exact 32,595-node graph and local algebra on a
T4. Gathered FP16 `SDPBackend.EFFICIENT_ATTENTION` at query chunk 2,048 used
1,057,530,880 peak allocated bytes and 143.412 ms forward+backward. The current
explicit FP32 chunk-8,192 path used 1,669,328,384 bytes and 129.253 ms. The
selected SDPA row therefore reduces isolated allocation by 36.65% for a 10.95%
time cost and has accepted output/gradient evidence.

Spec 0031 FlexAttention remains unexecuted and is not selected. Direct analysis
of its real WSI45630 block plans shows 6,153,388 candidate pairs for block 16
and 12,257,692 for block 32, versus 825,676 exact patch-plus-null pairs. The
common null block also widens transpose metadata across every query block.
FlexAttention is block sparse rather than edge sparse; this graph is a fixed
degree-26 workload, so gathered SDPA is the stronger already-measured primitive.

## Preserved Architecture And Algebra

- Width 192, six heads of width 32, two local blocks, one CLS plus 16 REGs,
  one calibrated-sigmoid all-patch read, one CLS-only 17-token softmax read,
  width-256 SwiGLU, five raw logits and class order `CC, EC, HGSC, LGSC, MC`.
- Each patch attends exactly its at-most-25 physical radius-two neighbours plus
  one learned null key with fixed zero value. Padding is masked to negative
  infinity and receives no mass. No absolute coordinates or global-to-patch
  path is introduced.
- Radial codebook `[0,1,2,4,5,8]`, learned per-head radial bias, learned null
  key and learned null bias are unchanged.
- Stored posterior `mu` remains immutable FP32. Only the CUDA-resident training
  copy is converted directly to FP16 before the model call; no persistent
  latent artifact changes.

## AMP Precision Contract

AMP-first does not mean `model.half()`.

- Model parameters, gradients, AdamW moments, normalization affine parameters
  and GradScaler state remain FP32.
- The CUDA-resident latent bag is FP16. Eligible convolutions, linear layers and
  attention kernels use the dtype selected by CUDA autocast.
- Every GroupNorm and LayerNorm is the unmodified PyTorch module and executes
  inside the enclosing autocast region. There is no custom wrapper, disabled
  autocast scope, forced norm dtype or requirement that norm outputs remain
  FP16. PyTorch owns the numerically appropriate casts and downstream eligible
  operations return to lower precision automatically.
- Spec 0034 version 1 failed before capacity measurement because the removed
  wrappers disabled autocast around FP16 inputs with FP32 affine tensors. That
  failed optimization is not part of the model contract.
- The calibrated global sigmoid core remains FP32 from QK dot through
  `beta-log(N)`, sigmoid, multiplication by V and the key-axis sum. At
  `N=32,595`, its zero-logit gate is about `3.07e-5`, below FP16's normal range,
  and its unnormalized learned sum has no safe FP16 magnitude bound.
- The calibrated sigmoid reduction output must not be cast back to FP16. Its
  learned, unnormalized evidence sum can exceed FP16's finite range even while
  remaining finite in FP32. The summary output projection, 17-token global
  residual/SwiGLU, CLS-only attention/SwiGLU, final LayerNorm,
  `Linear(192,5)`, raw logits and cross-entropy all remain FP32. Their footprint
  is negligible relative to the patch stream.
- Training uses `torch.autocast("cuda", dtype=torch.float16)` and
  `torch.amp.GradScaler("cuda", init_scale=32768, growth_interval=1_000_000)`;
  backward runs outside autocast.
- On overflow, GradScaler skips that optimizer update, lowers its scale and the
  data iterator advances to the next WSI. The skipped WSI is not retried or
  requeued at epoch end, and an update-based LR scheduler does not advance.
  Record skipped attempts by classifier and diagnosis; repeated or class-skewed
  skips invalidate the run as a stability problem rather than being hidden by
  retries.

## Fixed-26 Fused Local Attention

Project Q/K/V once for all `N` tokens. For each fixed query chunk of 2,048:

```text
q              [C,6,1,32]
gathered k/v   [C,6,25,32]
append null    [C,6,26,32]
additive bias  [C,6,1,26] FP16
```

The first 25 slots use the canonical graph arrays. Invalid slots use a safe
index but are masked; slot 26 is the single valid null. Invoke
`scaled_dot_product_attention` with `dropout_p=0`, `is_causal=False`, and
`scale=1/sqrt(32)`, and force `SDPBackend.EFFICIENT_ATTENTION` on CUDA with no
silent backend fallback. CPU tests use the math backend. The measured T4 kernel
accepts FP16 inputs at head width 32 and passed the Spec 0028 behavioral
output/gradient gate; no undocumented internal accumulation layout is treated
as a stable contract.

Packing is required and uses these exact logical slices:

- local Q/K/V: one bias-free `Linear(192,576)`, ordered Q,K,V;
- each SwiGLU input: one biased `Linear(192,512)`, ordered gate,value;
- global summary: separate Q plus one packed bias-free `Linear(192,384)`,
  ordered K,V;
- final CLS read: separate Q plus one packed bias-free `Linear(192,384)`,
  ordered K,V.

Initialize each logical slice independently, in the listed order, through the
active construction RNG exactly as an unpacked layer would be initialized;
initialize packed biases to zero. Packing changes kernel launch/layout only,
not logical parameters or algebra.

CUDA uses fixed 2,048-query inner chunks and pads only the final query chunk
before SDPA, then slices padded queries away before the output projection. For
WSI45630 this final chunk contains 1,875 real and 173 padded queries. Every real
patch is processed exactly once; padded rows must contribute neither downstream
output/gradient nor input/parameter gradient.

Each normal-VAE-latent and SO(2)-VAE-latent classifier owns an independent
training-step callable, optimizer, GradScaler, checkpoint cadence and stopping
time; neither update waits for or gates the other. The runner compiles each
model's heavy numerical path and its fused AdamW step independently. AOTAutograd
generates the corresponding backward graphs. `fullgraph=False` is permitted at
the top-level step because the local chunk dispatcher is the one explicit eager
control-flow boundary; the runner must still report every other graph break,
recompile and eager fallback.

Do not use blanket `dynamic=True`. Before the first compiled invocation, mark
only dimension zero of the device latent tensor `[N,16,32,32]` and the three
device graph tensors `[N,25]` with `torch._dynamo.mark_dynamic(..., 0, min=1,
max=32595)`, then use the default `dynamic=None` compiler policy with
`mode="max-autotune-no-cudagraphs"`. The compiler must keep latent channels,
spatial dimensions, token/head widths, neighbour width, global-token count and
class count static. A strict mark is intentional: specialization of a declared
patch-count axis is a failure, not a silent per-bag compile.

A local PyTorch 2.13 strict `fullgraph=True` Dynamo trace tried the complete
model first, with exactly those four axis-zero annotations. The patch encoder
accepted two distinct `N` values in one graph, but the complete model failed at
`range(0, N, 2048)` in `_local_query_chunks`: Dynamo specialized `N`, exactly as
the strict mark is designed to expose. `torch.while_loop` cannot repair this
because its current API does not support training. The selected fallback is
therefore the smallest possible scope reduction: keep only that Python chunk
dispatcher eager; compile the dynamic-`N` patch projections/global path and the
fixed 2,048-query attention core. The sigmoid cardinality term now constructs
`N` as a device FP32 scalar before `log`, and a strict two-size compiled test
proves the global summary no longer specializes by bag size. Whole-model
dynamic FlexAttention compilation remains outside this implementation because
FlexAttention is not the selected attention backend.

## Hot-Path And Runtime Contract

- Authenticate and hash the canonical graph once on CPU before either branch
  loads latent values. Device copies are trusted derivatives of that verified
  object. Model forward performs no `.cpu()`, `.tolist()`, hash, file read or
  host synchronization.
- Keep `cudnn.benchmark=True` and `cudnn.deterministic=False` in the later CUDA
  runner.
- A later training runner uses explicit `AdamW(..., fused=True)` with the
  existing semantic decay groups and attempts a separately compiled optimizer
  step for each model. Eager fused AdamW is the measured fallback. `foreach` is
  not an implicit substitute.
- Keep `optimizer.zero_grad(set_to_none=True)`, FP16 autocast cache enabled,
  channels-last convolution weights and device bags, and nondeterministic
  algorithms enabled for the speed-first runner. DDP-only Python-reducer,
  communication-hook and compiled-autograd-overlap settings do not apply: the
  two models live independently on the two T4s and do not synchronize gradients,
  optimizer updates, checkpoints or termination.
- Do not select a CUDA-graph compile mode before capacity evidence; both
  `reduce-overhead` and ordinary `max-autotune` may retain extra device memory.
- Activation checkpointing is not selected pre-emptively because it recomputes
  work. A later capacity probe may add non-reentrant checkpointing only if this
  AMP-first path still lacks the precommitted VRAM headroom.

## Non-Goals

- No architecture width/register/radius/activation change, custom Triton
  kernel, FlexAttention selection, patch reordering, background tokens,
  sampling, learning run, test access, dataset publication, Kaggle push, paper
  claim or issue update.
- No claim of full-model capacity or speed from CPU tests or isolated Spec 0028
  measurements.
- No blanket FP16 conversion of normalization, calibrated sigmoid aggregation,
  final logits/loss, parameters, gradients or optimizer state.

## Local Acceptance Criteria

1. Exact model parameter count remains 1,513,055 and the logical architecture
   manifest remains unchanged except for precision/backend/packing.
2. Independent fixed-26 oracle tests cover holes, boundaries, row-wrap traps,
   all radial codes, nonzero radial/null parameters, padding exclusion, outputs,
   input gradients and every logical parameter gradient.
3. Packed projection slices have the intended independent initialization and
   reproduce an unpacked reference forward, input/logical-parameter gradients,
   and one optimizer update.
4. Structural tests require every GroupNorm and LayerNorm to be an unmodified
   PyTorch module. No local norm wrapper or manual dtype override is permitted;
   the later T4 gate records the dtypes selected by CUDA autocast.
5. A final-chunk test proves exact real-query equivalence, exactly one visit per
   real patch, and zero padded-row contribution after slicing.
6. Graph integrity is checked explicitly before transfer, but model forward has
   no integrity rehash or host synchronization.
7. Focused tests, `./scripts/python_quality.sh`, both agent preflights and
   `git diff --check` pass. Two clean-context reviews find no P0/P1 blocker.

## Later CUDA Capacity Gate

A new spec and fresh authorization must package the exact model. A bounded
hostile/small-bag CUDA equivalence phase must first trace dtypes, prove the fused
efficient backend, and compare output and every gradient to an FP32 oracle.
Output tolerances are `rtol=atol=5e-3`; every nontrivial input and parameter
gradient requires relative L2 error at most `2e-3`, cosine similarity at least
`0.999`, and nonzero norm in both paths. The separate exact 32,595-patch phase
is candidate-only because the FP32 reference is already known not to fit; it
must instead prove finite loss/gradients, complete parameter-gradient coverage,
and the exact 1,875-real/173-pad final chunk.

Each full-bag run must use fail-closed `SDPBackend.EFFICIENT_ATTENTION` and
`AdamW(fused=True, lr=2e-4, betas=(0.9,0.999), eps=1e-8)` with the semantic
decay groups from `local_global_mil_adamw_parameter_groups`: ordinary matrix
weights use `5e-3`; global CLS/REG tokens, local null keys, relative-bias tables,
all 1D offsets, biases, and norm scales use zero. Backend records
must prove no SDPA, foreach, or single-tensor fallback. Record compile-startup
and settled forward/backward/step peaks with optimizer state resident,
full-step graph breaks/recompiles/eager fallbacks, and complete at least two
optimizer steps independently for each model. A failure or GradScaler skip in
one run must be recorded but must not block or roll back the other; the affected
WSI attempt is consumed without an optimizer/scheduler step and is not requeued.
Spec 0030's
consumed authority and failed artifact cannot be reused.

## Local Acceptance Evidence

Accepted on 2026-09-02: 33 focused Spec 0026 model/graph/attention tests pass;
the full quality gate passes with 1,095 tests and one expected GPU-only skip;
Ruff, BasedPyright and formatting are clean. Three independent adversarial
reviews found no P0/P1 blocker. This is CPU/local evidence only and does not
claim revised full-bag T4 capacity or throughput.

## Primary References

- PyTorch AMP: https://docs.pytorch.org/docs/stable/amp.html
- PyTorch SDPA: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch FlexAttention: https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html
- FlexAttention/FlashAttention-4 post: https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- PyTorch compile: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch dynamic-shape annotations: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- PyTorch `while_loop`: https://docs.pytorch.org/docs/main/higher_order_ops/while_loop.html

## Related Files

- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/specs/0028-wsi45630-local-softmax-repair-probe.md`
- `docs/specs/0030-local-global-mil-capacity-probe.md`
- `docs/specs/0031-coordinate-flexattention-probe.md`
- `src/eqvae/models/local_global_mil.py`
- `tests/test_spec0026_attention.py`
- `tests/test_spec0026_mil_model.py`
