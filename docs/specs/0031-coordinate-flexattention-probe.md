# Spec 0031: Coordinate-Masked FlexAttention Probe

Status: superseded without CUDA execution by Spec 0032
Implementation readiness: retained as rejected-candidate evidence; do not launch
Owner/workstream: Spec 0026 local-attention capacity repair
Last updated: 2026-09-02

## Purpose

Test whether PyTorch FlexAttention can execute the exact Spec 0026 local
softmax/null algebra without gathered neighbour tensors. The candidate must use
physical WSI coordinates, exclude removed background patches, preserve the
learned radial bias and single null sink, and reduce T4 memory enough to justify
replacing the explicit FP32 backend.

Spec 0032 supersedes this launch path. Exact graph analysis found 7.45x/14.85x
candidate-pair overcomputation for block sizes 16/32, while Spec 0028 already
measured a lower-memory exact fixed-26 efficient-SDPA path on the T4. This
candidate remains useful implementation evidence but is not an active probe.

## Non-Goals

- No change to width 192, six width-32 heads, two local blocks, CNN channels,
  global attention, classifier, labels, splits, patch membership or ordering.
- No dense `N x N` mask or score tensor, sequence-adjacency neighbourhood,
  background placeholder tokens, approximation of the null sink, learning,
  validation/test access, publication or paper claim.
- No Kaggle push, dependency publication or full-model retry without fresh
  explicit authorization.

## Inputs And Data Contract

- Reuse only the private Spec 0027/0028 coordinate dataset
  `maximusshtefan/eqvae-wsi45630-capacity-inputs`, immutable version 1.
- Authenticate the existing contract and pointer hashes before graph use.
- Use all 32,595 unique WSI45630 coordinates in their existing numeric `(y,x)`
  order and generated FP16 tokens `[32595,192]` with seed 1701.
- Physical lattice coordinates are `g_i=(x_i/256,y_i/256)`. Sequence positions
  never define adjacency.
- The exact patch edge predicate is
  `abs(gx_i-gx_j) <= 2 and abs(gy_i-gy_j) <= 2`. Self is included; absent
  coordinates have no Q/K/V token and no edge.

## Attention Contract

Use six heads of width 32 and the existing independent Q/K/V plus biased output
projections. Append one learned key and one fixed zero value to K/V only:

```text
patch_score_ijh = dot(q_ih,k_jh) / sqrt(32) + B_h[r2_ij]
null_score_ih   = dot(q_ih,k_null_h) / sqrt(32) + gamma_h
weights         = softmax(valid local patch scores + one null score)
output          = OutProj(sum(weights * values))
```

`B[6,6]`, `k_null[6,32]` and `gamma[6]` are trainable and initialize to zero.
The squared-radius codebook remains `[0,1,2,4,5,8]`. A missing patch is masked
at the score/edge level; no token feature may contain `-inf`.

## FlexAttention Representation

- Q has the `N` packed foreground tokens. K/V have those same `N` tokens plus
  the one null entry. Holes never become tokens.
- Construct a `BlockMask` from the canonical sparse graph. For each query tile,
  list only KV tiles containing one of its true radius-two neighbours plus the
  null tile. A `mask_mod` rejects non-neighbour pairs inside partial tiles.
- A `score_mod` adds the learned radial bias for patch edges and `gamma_h` for
  the null edge. Captured coordinate tensors are representation metadata; only
  the radial/null parameters require gradients.
- `create_block_mask` is forbidden at full size. Build both forward
  `(kv_num_blocks,kv_indices)` and transpose `(q_num_blocks,q_indices)` metadata
  directly from the canonical graph, then construct `BlockMask` with
  `seq_lengths=(N,N+1)`. Metadata may use rectangular block-level padding but
  must remain below 32 MiB and record every tensor shape/byte count.
- Test query/KV block sizes 16 and 32 unless the installed T4 backend rejects
  one. Force `BACKEND="TRITON"`, compile the Flex call with `fullgraph=True` and
  reject unsupported execution; AUTO, uncompiled execution and the unfused
  dense debug implementation are forbidden in full-graph measurement.
- Q/K/V projections may run in FP16 as in Spec 0026, but cast their outputs to
  FP32 before FlexAttention. The dot product, radial/null addition, softmax and
  value reduction remain FP32 with CUDA autocast disabled; cast the context
  back only before the output projection.
- Record construction/compile time separately. Selection uses settled
  forward/backward time and peak allocated/reserved VRAM; compile time is not a
  cost for the later learning run.

## Correctness And Measurement Contract

Before full-graph timing, use an irregular hostile graph containing row-wrap
adjacent sequence entries, holes, boundary nodes, every radial code,
deliberately nonzero radial/null parameters and a fixed upstream gradient.
Compare FlexAttention against the independently assembled per-query oracle and
the accepted explicit implementation.

- Output: finite and elementwise `rtol=5e-3, atol=5e-3`.
- Each input/parameter gradient: finite, relative L2 at most `2e-3`, cosine at
  least `0.999`, with nontrivial oracle norm.
- Require nontrivial oracle gradients for every radial-table column in every
  head and every component of the null parameters.
- Require exact mask equality for every `(query,key)` pair inside every listed
  partial tile. For coordinate-far consecutive sequence entries, run
  bidirectional query-local Jacobian probes: upstream supported only on query
  `i` must give excluded token `j` exactly zero gradient, and vice versa.
- On the complete 32,595-token graph, compare the full output, input gradient
  and every parameter gradient against explicit attention using the same gates.
- Run two warmups and five measured forward/backward iterations for every
  supported correct full-graph row. Record mean/min/max time and peak VRAM.
- The current explicit FP32 chunk-8192 row remains the measured reference. A
  Flex row may only nominate a later full-model retry when it is correct, uses
  at most 1.10 GiB peak allocated memory, reduces the freshly measured explicit
  peak by at least 35%, and takes at most twice the explicit mean time. Fastest
  mean time selects between qualifying Flex rows.

## Outputs And Acceptance Artifacts

- Standalone private-kernel source and metadata under
  `kaggle/kernels/wsi45630_flex_attention_probe/`.
- Focused CPU tests exercise graph-to-block conversion, exact masking, radial
  and null algebra, row-wrap rejection, source size and account-portable
  metadata. Small CPU FlexAttention may use the documented unfused debug path
  only as a functional oracle; it is never a performance result.
- A future authorized Kaggle artifact records runtime identity, graph identity,
  Torch/Triton versions, forced backend/kernel options, compile counters,
  block-index shapes/counts/bytes, correctness metrics, timing, VRAM, failures
  and an explicit nomination decision.
- Spec 0026 and the reusable model remain unchanged until accepted CUDA
  evidence nominates FlexAttention and a separate exact two-block full-model
  capacity run passes with recorded headroom and settled step time.
- Future private metadata uses slug `eqvae-wsi45630-flex-attention-probe`, with
  exactly `maximusshtefan/eqvae-wsi45630-capacity-inputs` as its sole source;
  competition/kernel/model source arrays remain empty. The generic portable
  path may rewrite only the top-level kernel owner to the authenticated actor.

## Acceptance Criteria

1. The coordinate predicate reproduces every edge in the canonical Spec 0026
   graph and rejects sequence-near/coordinate-far pairs.
2. No implementation path allocates or serializes an `N x N` object.
3. CPU-focused tests, `./scripts/python_quality.sh`, both preflights and
   `git diff --check` pass.
4. Require Tesla T4 / SM75, bootstrap the latest Torch stack before its first
   import, and fail closed unless compiled Triton FlexAttention is evidenced.
5. Two independent clean-context reviews find no P0/P1 scientific,
   numerical, memory, provenance or launch-safety blocker before any push.
6. Remote execution requires a new one-shot authorization guard, durable claim
   before the network call, accepted canonical actor/slug/version receipt, and
   non-overwriting exact-version JSON/log retrieval with per-file hashes. A
   failed or ambiguous attempt consumes that authority.
7. Kernel evidence may only nominate a separately authorized full-model
   capacity retry; only that two-block result may change Spec 0026's backend.

## Local Implementation Evidence

- Candidate source SHA-256:
  `6a79b9c4b7555120e98e8783704090f7f990913927e1b2d22e1d90994cd48989`.
- Metadata SHA-256:
  `527a4e9bc9a0bb1a30d76da872c9a0e097457ccae14f3d99e470ed4f20f68032`.
- Eight focused tests pass. The complete repository suite passes with 1,083
  tests and one skip; whole-repo Ruff and basedpyright pass with zero errors.
  Both preflights and `git diff --check` pass.
- Two independent final adversarial reviews found no P0/P1 scientific,
  numerical, memory, provenance or launch-safety blocker.
- The generic Kaggle push path rejects both this kernel directory and immutable
  kernel ID. A dedicated one-shot claim/receipt/retrieval route is intentionally
  absent until a fresh remote authorization exists.
- No CUDA result exists. These local checks do not establish T4 support,
  performance, memory reduction or fitness of the full two-block MIL model.

## Known Risks And Adversarial Checks

- Block tiling may compute many rejected pairs inside partial blocks and lose
  the expected memory/time benefit; measure rather than infer from asymptotics.
- FlexAttention is a prototype API. Reject silent dense fallback, graph breaks
  that materialize scores, unsupported learned-bias backward and nonfinite
  gradients.
- The null KV block is globally listed but carries a fixed zero value; verify
  that this is one sink per query, not one sink per missing coordinate.
- Coordinate tensors must remain branch-identical and derived before either
  latent payload is read in later full-model use.

## Implementation Blockers

None for local implementation. CUDA/T4 execution and any backend replacement
remain blocked on fresh remote authorization and accepted evidence.

## Related Files

- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/specs/0028-wsi45630-local-softmax-repair-probe.md`
- `docs/specs/0030-local-global-mil-capacity-probe.md`
- `src/eqvae/models/local_global_mil.py`
- `CURRENT.md`
