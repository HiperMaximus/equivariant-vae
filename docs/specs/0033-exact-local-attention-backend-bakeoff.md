# Spec 0033: Exact Local-Attention Backend Bakeoff

Status: native candidates locally verified / private CUDA probe authorized
Implementation readiness: exact account-portable T4 probe may launch once
Owner/workstream: Spec 0032 compiler and fixed-degree-kernel follow-up
Last updated: 2026-09-02

Launch authority: `spec0033_inductor_attention_probe_authorized`. The user
authorized the exact private package on 2026-09-02. This authority covers one
probe launch with the shared owner-qualified WSI45630 input; it does not cover
learning, sealed-test access, or a production-backend change.

## Purpose

Remove the Python query-chunk dispatcher if an exact, lower-memory and fast
Tesla-T4 implementation exists. Preserve the Spec 0026/0032 MIL architecture
and compare execution backends before changing the selected model.

## Preserved Contract

- Width 192, six heads of width 32, two local blocks, radius-two physical
  lattice graph, at most 25 present patches per query, one learned null key
  with fixed zero value, six learned per-head radial biases, and local softmax.
- Missing tissue coordinates and coordinates outside the slide are absent keys,
  not zero-valued image tokens. Sequence adjacency never defines an edge.
- FP32 master parameters and AMP FP16 patch-path tensors remain unchanged.
- No candidate may allocate, materialize or serialize an `N x N` tensor.
- The current 2,048-query fixed-26
  `SDPBackend.EFFICIENT_ATTENTION` implementation remains the fallback until a
  complete T4 gate selects a replacement.

## Candidate Funnel

1. **Measured fallback:** current gathered fixed-26 efficient SDPA with the
   Python chunk dispatcher.
2. **Whole-bag fixed-25 Inductor:** native tensor gather, FP32 score/softmax
   opmath, explicit null denominator and weighted-value reduction over the
   static 25 slots. Compile with dynamic `N` and `fullgraph=True`.
3. **Edge-segment Inductor:** flatten only valid graph edges, compute edge
   scores, stable row softmax with `torch.segment_reduce`, and row-reduce
   weighted values. The CSR row/column/radial arrays are prebuilt and passed as
   tensors; graph construction is never in the compiled step.
4. **Custom Triton:** only if candidates 2 and 3 fail correctness, capacity or
   settled throughput. Use `torch.library.triton_op`, `wrap_triton`, registered
   autograd and a CPU/native reference. Do not begin this maintenance-heavy arm
   merely because it is theoretically fusible.

Test the two native candidates under:

- `mode="default"`;
- `mode="max-autotune-no-cudagraphs"`;
- that mode plus experimental pointwise/combo-kernel autotuning when the
  installed PyTorch exposes the options.

Keep default shape padding, epilogue fusion, peak-memory reordering, in-place
buffers and persistent reductions enabled. Do not enable CUDA graphs for the
dynamic bag axis. Compile startup is unscored; settled time per training step is
the speed metric.

## Existing-Implementation Audit

- NATTEN FNA is the closest dense 2-D kernel and supports Turing plus additional
  K/V tokens, but its current operation API has no arbitrary tissue-hole mask
  or learned radial score-bias input. Its shifted boundary neighborhoods also
  differ from truncated physical neighborhoods.
- xFormers and FlashAttention local masks describe contiguous sequence windows
  or supported block patterns. A custom irregular mask becomes dense or needs a
  new kernel; sequence-local windows are incorrect for row wraps and holes.
  Mainline FlashAttention-2 does not target Turing/T4.
- PyG can compile custom message passing, but stock `TransformerConv` adds edge
  features to keys and values rather than applying this scalar score bias. A
  custom PyG layer is the same edge-softmax/scatter algorithm as candidate 3
  with another dependency.
- DGL edge softmax expresses candidate 3, but current official binaries do not
  match the selected PyTorch 2.13/CUDA 13 environment.
- Native sparse CSR `sampled_addmm` computes sampled scores, but CSR sparse
  softmax is unavailable; CSR-to-COO sparse softmax remains outside a strict
  compiled graph. It is excluded.
- FlexAttention remains excluded by Spec 0031's measured real-graph tile
  overcomputation.

These exclusions concern exact replacement of the locked algebra. They do not
claim the libraries are generally slow or unsuitable for other architectures.

## Dynamic Compilation And Training Step

- Mark only dimension zero of `[N,16,32,32]`, `[N,25]`, and edge arrays as
  dynamic with bounds `1..32595`; keep all feature, head and degree axes static.
- First require each local kernel to compile `fullgraph=True` and execute two
  distinct graph sizes without recompilation or eager fallback.
- Once a native candidate passes, attempt the full model-plus-loss as one
  dynamic compiled graph. Use AOTAutograd for its generated backward and test
  experimental Compiled Autograd as a measured candidate.
- Compile `AdamW.step()` separately with `fused=True`; keep
  `zero_grad(set_to_none=True)` outside the timed compiled numerical regions.
  Normal- and SO(2)-latent classifiers retain independent callables,
  optimizers, scalers and termination.
- `torch.compiler.nested_compile_region` may be tested to reduce cold compile
  time for the two structurally identical local blocks, but it cannot win a
  settled-runtime comparison because it is a compile-time mechanism.
- Reject unsafe guard skipping. Experimental numerical/compiler features are
  otherwise eligible when feature-detected and bounded correctness passes.

## Correctness And Measurement Gate

Use the hostile small graph and exact WSI45630 graph already owned by Specs
0027-0032. Initialize nonzero radial/null parameters and use a fixed upstream
gradient.

1. Compare outputs, input gradients and every logical parameter gradient to the
   independent FP32 fixed-26 oracle. Use the Spec 0032 tolerances: output
   `rtol=atol=5e-3`; nontrivial gradients relative-L2 at most `2e-3` and cosine
   at least `0.999`.
2. Prove exact zero influence for coordinate-far sequence-adjacent patches,
   exact masking of holes/padding, all radial codes, and one null per query.
3. Record graph breaks, unique graphs, recompiles, generated kernel count,
   backend/options, Torch/Triton/CUDA identity, settled forward/backward time,
   and peak allocated/reserved VRAM.
4. Inspect generated code or compiler traces. Reject a nominally compiled
   candidate that falls back eagerly or materializes an `N x N` object.
5. Run at least two full optimizer steps with optimizer state resident. A
   replacement must fit the complete two-local-block model on T4 with at least
   512 MiB unallocated headroom and beat the fallback's settled step time; if
   only one condition improves, keep the fallback and report the tradeoff.

## Outputs And Acceptance Artifacts

- Native candidate functions and CPU correctness/strict-compile tests.
- The account-portable private Kaggle probe package and immutable result under
  the launch authority above.
- A compact machine-readable candidate table with rejection reasons.
- Update Spec 0032 and the production model only after accepted T4 evidence.

## Non-Goals

- No change to model width, heads, registers, FFN, attention activation,
  positional/radial scheme, graph radius, patch sampling, learning schedule,
  dataset split, classifier evaluation or paper claims.
- No learning run, sealed-test access, or production-backend change is
  authorized by this spec.
- No dense-grid imputation or sequence reordering may alter graph semantics.

## Local Acceptance Criteria

1. Both native candidates match the oracle in eager mode and pass backward.
2. Both candidates are attempted under strict dynamic two-size compilation;
   unsupported operations are recorded rather than hidden by graph breaks.
3. Focused package/model tests, touched-file lint/type checking, kernel
   validation and `git diff --check` pass. Repository-wide tests are not a
   launch gate for this disposable probe.
4. Independent clean-context review is attempted; if agent capacity is
   unavailable, record that blocker and do not claim reviewed acceptance.

## Local Evidence

- Both native candidates match the independent FP32 oracle in forward and all
  Q/K/V, radial-bias, null-key and null-bias gradients on a hostile graph.
- Each candidate passed CPU Inductor `fullgraph=True` with strict dynamic node
  and edge axes across two distinct sizes using one unique graph.
- A state-compatible fixed-25 replacement preserved the locked model state and
  eager logits. The complete CNN, two local blocks, global summary, CLS read,
  classifier and generated backward then passed CPU Inductor
  `fullgraph=True` for two distinct bag sizes with one unique graph. This
  confirms the Python dispatcher was the remaining full-model graph break.
- Generated-code inspection found that the fixed-25 forward fused indexed K/V
  loads, scores, radial mask/bias, null, softmax and value reduction into one
  kernel. It retained compact `[N,6,26]` softmax state and `[N,6,32]` output,
  without fixed-slot `[N,25,6,32]` K/V or message buffers.
- The edge-segment forward kept `segment_reduce` as external calls and
  materialized `[E,6,32]` FP32 weighted messages. At WSI45630's 793,081 edges
  this alone is about 581 MiB; valid edges occupy 97.3% of the fixed 25 slots.
  Keep it as a graph-library-pattern control but prioritize fixed-25 on T4.
- The final launch gate passed 16 focused package/candidate/launcher tests,
  touched-file Ruff/BasedPyright, shell syntax, kernel validation and
  `git diff --check`. A repository-wide run completed before the user narrowed
  disposable-probe policy: 1,104 passed and one expected GPU-only skip. Three requested
  Sol-high clean-context reviews could not start because the shared agent
  account exhausted its usage allocation, so independent review remains
  explicitly incomplete. The private account-portable probe completed as
  `maximshtefan/eqvae-wsi45630-inductor-attention-probe/1`; its authenticated
  report is
  `runs/kaggle/spec0033_inductor_attention_probe_v1/spec0033_inductor_attention_probe.json`
  (SHA-256 `af34b43ddb29c23e0c6debdcb0e42e671729a6abd56cee61d067c4f402c0aaa2`).
- On PyTorch 2.14.0+cu130 / Tesla T4 and the exact 32,595-node, 793,081-edge
  graph, fixed-25 `max-autotune-no-cudagraphs` matched the oracle, retained
  finite gradients, and used one graph with zero graph breaks. It measured
  17.87 ms and 276 MiB peak allocated versus chunked eager SDPA's 151.51 ms and
  1,057 MiB: 8.48x faster and 73.9% less allocated memory. Default fixed-25 was
  18.43 ms; the aggressive experimental profile regressed to 24.16 ms.
- Edge-segment was correct but measured 27.35 ms and 1,384 MiB peak allocated,
  confirming it is inferior to fixed-25 for this dense radius-two graph.
- The CUDA report measures one isolated local-attention forward/backward. It
  does not yet satisfy the full two-local-block model plus optimizer/headroom
  acceptance gate, so it selects the next candidate but does not by itself
  authorize replacing the production backend.

## Primary References

- PyTorch compile: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch dynamic shapes: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html
- PyTorch Compiled Autograd: https://docs.pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
- PyTorch user Triton kernels: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- PyTorch regional compilation: https://docs.pytorch.org/docs/stable/generated/torch.compiler.nested_compile_region.html
- NATTEN operations: https://natten.org/operations/
- PyG TransformerConv: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html
- DGL edge softmax: https://www.dgl.ai/dgl_docs/_modules/dgl/ops/edge_softmax.html
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- xFormers attention biases: https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/attn_bias.py

## Related Files

- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/specs/0031-coordinate-flexattention-probe.md`
- `docs/specs/0032-amp-fixed26-mil.md`
- `src/eqvae/models/local_global_mil.py`
- `tests/test_spec0026_attention.py`
