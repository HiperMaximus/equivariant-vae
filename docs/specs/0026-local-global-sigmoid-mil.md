# Spec 0026: Local-Global MIL

Status: implemented / local acceptance complete
Implementation readiness: local model complete; capacity execution, learning, and
test access remain unauthorized
Owner/workstream: full-foreground WSI diagnosis architecture
Last updated: 2026-09-02

## Purpose

Replace the capacity-only width-128 all-token Transformer with one nearly fixed MIL
architecture designed for the hypothesized local/global structure of ovarian
diagnosis: patches exchange only local spatial context, while a small learned
stream reads the complete WSI for classification.
The design must consume Spec 0025's complete foreground bags without quadratic
patch attention, preserve a controlled normal-VAE versus SO(2)-VAE comparison,
and remain small enough for one complete WSI per T4.

This spec locks one architecture. Local softmax with one static zero-value null
slot is the chosen design. Spec 0028 validated that algebra on the real
largest-bag graph; after Spec 0030 exposed a full-model OOM, Spec 0032 selected
the already-measured FP16 efficient-SDPA row, fixed query chunk 2,048, AMP
precision boundaries and packed logical projections. It did not compare local
softmax against local sigmoid and therefore does not establish an
activation-quality winner. This spec does not define a learning ablation
matrix. The earlier Spec 0023 Transformer remains immutable capacity evidence for its own
executed bytes; it is not the architecture selected here.

## Non-Goals

- No Kaggle dataset publication, kernel push, training run, test release, paper
  claim, or issue update.
- No VAE retraining, latent modification, patch sampling, bag truncation, mean
  pooling baseline, architecture grid, repeated seed, or generalized tuner.
- No unrestricted patch self-attention, global-to-patch feedback, anchored
  region tokens, absolute WSI positional embedding, dynamically updated local
  register tokens, or multi-scale branch.
- No claim that sigmoid patch-summary attention, final CLS softmax, 16 registers,
  or this architecture improves learning until comparative development evidence
  exists.
- No absolute claim that the classifier proves latent usefulness: the coordinate
  graph exposes bag geometry even though token values receive no absolute
  coordinates.

## Inputs And Data Contract

- Reader: `FullForegroundBagDataset` from Spec 0025, using only the six staged
  files inside its `development/` directory and the externally pinned
  `dataset.json` SHA-256.
- Splits: frozen 106 train and 23 validation WSIs. Test remains sealed.
- Statistical unit: one complete WSI bag with one diagnosis label.
- Bag tensor: FP32 posterior means `mu[N,16,32,32]`; `N` varies by WSI and the
  largest accepted bag has 32,595 patches.
- Patch identity: the reader's ordered `WSIInstance` sequence. Coordinates are
  integer WSI pixel positions `(x,y)` on the fixed non-overlapping 256-pixel
  patch lattice. Membership and ordering may not change by model branch.
- Controlled independent comparison: normal-VAE and SO(2)-VAE runs receive
  identical logical instances, graph edges, classifier initialization, initial
  optimizer/schedule policy, WSI order and validation access. Only latent payload
  values differ. Each run owns its optimizer, GradScaler, checkpoints, early
  stopping and finalization; neither waits for, gates or rolls back the other.
- VAE parameters remain frozen; posterior sampling is forbidden.

Before graph construction, require `x % 256 == 0`, `y % 256 == 0`, unique
coordinates within the bag, one WSI identity, and exact instance count. Define
`g_i=(x_i/256,y_i/256)`; never infer adjacency from CSV row number.

Build one representation-independent graph object from the ordered
`LoadedWSIBag.instances` before loading either model's latent values. It contains:

```text
neighbor_index [N,25]  int64, padded with -1
neighbor_valid [N,25]  bool
radial_code    [N,25]  uint8, padded with 255
```

Valid neighbours are sorted lexicographically by `(delta_gx, delta_gy)`; the
codebook order is squared radius `[0,1,2,4,5,8]`. Before device transfer, hash
the following exact framed little-endian representation. A frame is
`uint16_le(tag_byte_length) || UTF8(tag) || uint64_le(payload_byte_length) ||
payload`. Concatenate these frames in the listed order and take SHA-256:

1. `schema`: ASCII `eqvae_spec0026_graph_v1`;
2. `wsi_id`: one signed `int64_le`;
3. `shape`: unsigned `uint64_le` values `(N,25)`;
4. `dtypes`: ASCII
   `coordinates:<i8;neighbor_index:<i8;neighbor_valid:u1;radial_code:u1`;
5. `sentinels`: signed `int64_le(-1)` followed by `uint8(255)`;
6. `codebook`: six consecutive `uint64_le` values `(0,1,2,4,5,8)`;
7. `coordinates`: ordered `(gx,gy)` pairs as contiguous row-major signed
   `int64_le` bytes with shape `[N,2]`;
8. `neighbor_index`: contiguous row-major signed `int64_le` bytes `[N,25]`;
9. `neighbor_valid`: contiguous row-major bytes `[N,25]`, exactly `0` or `1`;
10. `radial_code`: contiguous row-major `uint8` bytes `[N,25]`.

The tag and payload length fields make the hash preimage domain-separated and
unambiguous. Pass the same canonical graph identity and graph-array values to
both latent runs; rebuilding topology from latent values or model outputs is
forbidden.

The normal-versus-SO(2) result comprises two independently executed exploratory
trajectories under the same coordinate-aware classifier contract, not an
optimization-seed robustness claim. It can support only an incremental
comparison between latent representations under this shared geometry.
Development artifacts must report prediction/error association with bag size
and graph-degree summaries so geometry/cardinality shortcuts remain visible;
this is an audit, not an extra architecture arm.

## Outputs And Acceptance Artifacts

Local implementation work covered by this spec must produce:

- one reusable model module under `src/eqvae/models/`;
- focused CPU tests for topology, graph construction, attention algebra,
  initialization, directionality, and CLS-only equivalence;
- an exact model/config summary with parameter count and tensor shapes;
- no model-side softmax: training artifacts retain five raw logits.

The reusable model and graph code must contain no Kaggle username, dataset slug,
mount path or credential assumption. It consumes tensors, ordered `WSIInstance`
metadata and a prebuilt graph only.

A later execution spec must define the capacity package and artifact. Such an
artifact proves only fit and numerical execution; a still later, separately
authorized learning protocol must establish optimization behavior.

### Professor-account portability

The later capacity/training package must be reproducible from a GitHub clone by
a collaborator using their own Kaggle account. GitHub authentication is unrelated
to Kaggle authentication. Portability requires all of the following:

- the collaborator authenticates the Kaggle CLI with their own credentials;
- every private input dataset is shared with that Kaggle account before launch;
- the package builder accepts an explicit Kaggle kernel owner and kernel slug,
  generating metadata ID `<owner>/<slug>` with a title that resolves to the same
  slug;
- dataset sources remain explicit full `owner/dataset-slug` references and may
  still name the original dataset owner after sharing;
- runtime source discovery resolves authenticated mounted files from
  `/kaggle/input` and logical source records, never from a hard-coded username or
  workstation path;
- generated source/config/metadata and their guards contain no credential bytes;
- a read-only preflight verifies the authenticated CLI identity and access to
  every private source before any push is permitted.

This local implementation does not define or launch that remote package. Its API
and tests must keep the future package account-neutral, and the later execution
spec must behaviorally test generation for at least two distinct kernel owners.

## Architecture Contract

All dimensions, streams, graph topology and attention algebras below are locked.
Implement the two local blocks with the Spec 0028-selected explicit sparse
softmax/null path.

### Trainable encoder for frozen latents

The trainable patch encoder expands channels in stages so the largest spatial
activation does not immediately use width 192:

```text
mu [N,16,32,32]
  -> Conv2d(16,64,k=5,s=2,p=2,bias=False)
  -> GroupNorm(8,64) -> GELU
  -> Conv2d(64,128,k=3,s=2,p=1,bias=False)
  -> GroupNorm(8,128) -> GELU
  -> Conv2d(128,192,k=3,s=2,p=1,bias=False)
  -> GroupNorm(8,192) -> GELU
  -> AdaptiveAvgPool2d(1) -> flatten
  -> X0 [N,192]
```

There is no additional patch-token linear projection. A `16 -> 192` first
convolution is forbidden: at the largest bag its `[N,192,16,16]` activation is
1,602,109,440 values, about 2.98 GiB in FP16 before autograd storage. The
selected first convolution's `[N,64,16,16]` activation is 534,036,480 values,
about 0.995 GiB in FP16. Frozen describes the stored VAE posterior means and VAE
parameters, not this CNN: every patch-encoder parameter is trainable.

### Shared Transformer sublayer form

Every attention stage is pre-norm and uses separate residual additions for
attention and FFN:

```text
u = x + OutProj(Attention(Norm1(x)))
x = u + SwiGLU256(Norm2(u))
```

`SwiGLU256(z)` has this exact logical algebra; gate and value are emitted by one
packed `Linear(192,512)` with independently initialized row slices:

```text
gate  = Linear(192,256)(z)
value = Linear(192,256)(z)
hidden = SiLU(gate) * value
output = Linear(256,192)(hidden)
```

The inner width is deliberately `256 = 4d/3`, not a mandatory doubling. At
this width, one SwiGLU has about the same parameter count as that block's Q/K/V
and output projections. This retains token-wise nonlinear transformation while
avoiding an FFN-dominated classifier for only 106 training WSIs.

LayerNorm uses width 192 and epsilon `1e-5`. There is no post-norm, LayerScale,
stochastic depth, attention dropout, or FFN dropout in version 1. A final
LayerNorm is applied to CLS before classification.

### Global patch-summary sigmoid attention

The one-way patch-summary module uses six heads of width 32. Elementwise sigmoid
does not remove the reason to use heads: each patch pair receives six independent
scalar gates, one for each learned value subspace, rather than one scalar
controlling all 192 channels.

Q uses `Linear(192,192,bias=False)`; K and V use one packed
`Linear(192,384,bias=False)` whose two logical row slices are initialized
independently in K,V order. The output projection has a bias initialized to
zero. Each summary head has a learnable scalar offset `beta_h` initialized to
zero. Sigmoid weights are never renormalized to sum one.

For a query with `n_i` valid keys:

```text
score_ijh = dot(q_ih,k_jh)/sqrt(32) + beta_h - log(n_i)
weight_ijh = sigmoid(score_ijh)
output_ih = sum_j weight_ijh * v_jh
```

There is no structural or positional bias in this all-patch summary. At zero
content logits, each weight is `1/(n_i+1)` and total gate
mass is `n_i/(n_i+1)`, keeping initialization order one across variable bag
sizes. The fixed `-log(n_i)` term is not a zero-initialized learned bias.

Q/K/V and output projections may run under FP16 autocast. The attention core is
exactly FP32:

```text
q32, k32, v32 = q.float(), k.float(), v.float()
score32 = sum(q32 * k32, head_width) / sqrt(32)
score32 += beta.float() - log(N)
weight32 = sigmoid(score32)
context32 = sum(weight32[...,None] * v32, key_dimension)
output = OutProj(context32 cast to the projection/autocast dtype)
```

Thus dot products, bias addition, sigmoid, multiplication by V, and key-axis
reduction all execute and accumulate in FP32; casting an FP16 dot product to
FP32 afterward is non-conforming. The implementation must disable CUDA autocast
around this complete core; `.float()` alone does not prevent autocast from
downcasting a later matmul. The global initial weights can be below
FP16's normal range at large `N`. No standard softmax-only SDPA call may silently
replace this operation. CPU algebra tests do not replace a later CUDA-FP16 dtype
audit.

### Two local graph-attention blocks: explicit sparse softmax plus null

Build one symmetric per-WSI graph from coordinates. Patch `i` attends exactly
the valid patches satisfying:

```text
max(abs(gx_i-gx_j), abs(gy_i-gy_j)) <= 2
```

The self edge is included, so maximum degree is 25. Missing tissue-grid
positions remain absent. There is no cardinality bias. Represent adjacency
sparsely as at most 25 indices plus a validity mask per patch. Materializing an
`N x N` score/mask tensor is forbidden.

Each local layer owns a learned per-head radial relative-bias table indexed by
`(delta_gx**2 + delta_gy**2) in {0,1,2,4,5,8}`. Initialize every table entry to
zero. Radial distance makes the bias orientation-agnostic under square-lattice
rotations/reflections and translation invariant; it does not make the square
Chebyshev graph continuously `SO(2)`-equivariant. No absolute coordinate enters
a token.

Apply two complete local blocks:

```text
X1 = LocalBlock1(X0)
X2 = LocalBlock2(X1)
```

No CLS/REG token participates in either graph. Each local layer uses softmax
over exactly 25 graph slots plus one layer-owned, per-head static null key and
learned null bias. Invalid graph slots are masked to negative infinity; the
null value is fixed at zero. It is a parameter, never a patch-updated residual
token, and therefore cannot transmit information between localities. The null
key and null bias initialize to zero; the radial table also initializes to
zero.

Local Q/K/V use one bias-free packed `Linear(192,576)` with independently
initialized Q,K,V row slices. Under CUDA FP16 autocast, gathered Q/K/V, radial
bias, context and the patch residual remain FP16. Fixed query chunks of 2,048
invoke fail-closed `SDPBackend.EFFICIENT_ATTENTION` with 26 K/V slots, explicit
scale `1/sqrt(32)`, no dropout and no causal mask. Only the last chunk is padded
to 2,048 queries and sliced before output projection. Spec 0032 owns the exact
precision, padding, normalization and backend contract.

Spec 0028 version 2 authenticated the real 32,595-node, 793,081-edge graph and
independent oracle on a Tesla T4. Explicit, automatic SDPA and efficient SDPA
all passed the precommitted output/gradient gate; forced Flash was unsupported
with the additive mask. The previously selected explicit 8,192-query row took
129.253 ms and allocated 1,669,328,384 peak bytes. The now-selected FP16
efficient-SDPA row at chunk 2,048 took 143.412 ms and allocated 1,057,530,880
bytes: 36.65% lower isolated allocation for a 10.95% time cost. This selects an
implementation backend among the tested softmax paths, not an activation or
learning advantage over sigmoid. It validates one local module in isolation,
not two stacked blocks, the CNN or full-model capacity.

### One-way global summary block

Create one learned CLS and 16 learned free REG tokens:

```text
T0 = [CLS; REG_1; ...; REG_16]  # [17,192]
```

Initialize them independently from `Normal(0,0.02)`. They have no coordinates,
anchors, labels, or prescribed semantic roles.

All 17 tokens query all locally contextualized patches once:

```text
q = LayerNorm_query(T0)
kv = LayerNorm_patch(X2)
u = T0 + OutProj(SigmoidCrossAttention(Q=q,K=kv,V=kv,bias=-log(N)))
T1 = u + SwiGLU256(LayerNorm_ffn(u))
```

Query and patch LayerNorms are distinct affine modules. The operation updates
only `T`; there is no forward path in which patches query or receive global
tokens. Gradients from the WSI loss may still train the patch stream through K
and V, which is required and is not global-token feedback in the forward graph.

There is one patch-summary read, not two. No absolute/global coordinate or
anchored-register bias is supplied; the global stream receives morphology and
the local spatial context already encoded in `X2`.

### CLS-only global-token block

The final global interaction computes only the CLS output row because the other
16 output rows that a full final block would produce have no later consumer. It
uses normal softmax attention over the fixed 17-token summary sequence:

```text
c0 = T1[CLS]
z = LayerNorm_attention(T1)
q = z[CLS]
kv = z
u = c0 + OutProj(SoftmaxAttention(Q=q,K=kv,V=kv))
c1 = u + SwiGLU256(LayerNorm_ffn(u))
```

K/V include CLS and all 16 REGs. There is no null token, structural bias,
cardinality bias or learned head offset: a common `-log(17)` term would cancel
under softmax. Use standard scaled-dot-product attention with query length one,
no mask and zero dropout. With dropout disabled, this must equal the CLS row of
a full 17-token softmax-self-attention-plus-per-token-FFN block with identical
parameters. Do not compute and discard the other 16 rows.

### Classifier

```text
logits = FP32 Linear(192,5)(FinalLayerNorm(c1).float())
```

Disable CUDA autocast around both the final LayerNorm and classifier linear;
`.float()` on the input alone does not guarantee an FP32 linear under active
autocast.

The output order follows the frozen diagnosis indices in the logical dataset:
`CC, EC, HGSC, LGSC, MC`. The implementation may present human-facing names as
clear-cell, endometrioid, high-grade serous, low-grade serous, and mucinous,
but it may not reorder targets. Cross-entropy consumes raw logits; softmax is an
inference/reporting operation only.

## Initialization And Optimizer Compatibility

- Convolutions: Kaiming normal, `fan_out`, ReLU gain as the established
  GELU-compatible approximation.
- Linear weights: Xavier uniform, gain 1.0. The classifier head is not
  zero-weight initialized.
- GroupNorm/LayerNorm scale: one; all normalization bias: zero.
- Conv biases: absent. Q/K/V biases: absent. Output/FFN/head biases: present and
  zero.
- CLS/REG: independent normal values with standard deviation 0.02.
- Relative-bias tables and learned per-head attention offsets: zero.
- AdamW uses decay `5e-3` for ordinary convolution and linear matrix weights.
  Biases, normalization parameters, relative-bias tables, learned attention
  offsets, the 2D CLS/REG token table, and both 2D local null-key tables receive
  zero decay by semantic role; dimensionality alone must not place learned
  tokens or attention offsets in the decayed group.
- Construct one CPU initialization and load byte-identical independent copies
  into the normal and SO(2) branches.

## Config Contract

The implementation config must expose and pin these values; a capacity or
learning artifact records the fully resolved config:

| Key | Locked value |
| --- | --- |
| `latent_shape` | `[16,32,32]` |
| `patch_size_pixels` | `256` |
| `patch_encoder_channels` | `[64,128,192]` |
| `patch_encoder_kernels` | `[5,3,3]` |
| `patch_encoder_strides` | `[2,2,2]` |
| `group_norm_groups` | `8` |
| `token_width` | `192` |
| `attention_heads` | `6` |
| `head_width` | `32` |
| `local_layers` | `2` |
| `local_chebyshev_radius` | `2` |
| `local_max_degree` | `25` |
| `local_relative_bias` | `radial_squared_distance` |
| `global_registers` | `16` |
| `global_patch_reads` | `1` |
| `global_to_patch_feedback` | `false` |
| `local_attention_activation` | `softmax_with_static_zero_value_null` |
| `local_attention_backend` | `fixed26_fp16_efficient_sdpa` |
| `local_attention_chunk_size` | `2048` |
| `global_patch_summary_activation` | `sigmoid` |
| `global_patch_summary_cardinality_bias` | `negative_log_valid_keys` |
| `final_cls_attention_activation` | `softmax` |
| `final_cls_attention_mask` | `none` |
| `local_attention_cardinality_bias` | `none` |
| `global_patch_summary_head_offset_init` | `0.0` |
| `qkv_bias` | `false` |
| `attention_output_bias` | `true` |
| `ffn_kind` | `swiglu` |
| `ffn_inner_width` | `256` |
| `norm_order` | `pre_norm` |
| `layer_norm_epsilon` | `1e-5` |
| `dropout` | `0.0` |
| `layer_scale` | `false` |
| `classifier_classes` | `5` |
| `classifier_precision` | `float32` |

Do not expose alternate widths, register counts, local radii, attention
activations, positional schemes, or feedback paths in the one-off runner.
Private constructor helpers may remove repetition without creating a model zoo.

## Complexity And Capacity Boundary

Attention connectivity is:

```text
two local blocks:  O(2 * 25N)
global summary:    O(17N)
final CLS row:     O(17)
```

Projection/FFN work remains width-dependent (`O(N*d^2)`), and CNN activations
remain material; absence of `N^2` attention does not make early width free. The
expected learned parameter count is 1,513,055 with two layer-owned static null
keys/biases. Final implementation must derive and assert the selected count
rather than copying it into an unchecked report.

The sparse implementation must also avoid excessive padded-neighbour lifetimes.
At the largest bag, one FP32 `[N,25,6,32]` gathered tensor is 156,456,000 values,
about 597 MiB (626 MB), before autograd storage. Do not simultaneously retain
duplicate gathered Q/K/V, score, weight, and context tensors across both local
layers. Local tests must assert sparse shapes; the eventual CUDA probe must
record peak memory around each stage. Any memory fallback must use activation
checkpointing or true recomputation, not ordinary chunked forwards that retain
all CNN activations for backward.

The first remote execution, if separately authorized, is a capacity/numerical
probe on train WSI 45630 with all 32,595 foreground patches for both latent
runs. Each run proceeds independently on its own T4. Capacity packaging is not
implementation-ready under this architecture spec. Before it is built, a
focused revision or execution spec must pin the seed, optimizer/loss,
AMP/GradScaler state, warmup and measured steps, independent failure semantics,
runtime/bootstrap, mounted-source allow-list, artifact/failure schema, and
package guard.

That execution contract must try the exact direct complete-bag path first. If it
predeclares an activation-checkpointing or patch-encoder recomputation fallback,
it must first compare direct versus fallback on a small CUDA bag using locked
dtype-specific tolerances, outputs, input gradients, and every parameter
gradient. A reference that OOMs cannot establish equivalence after the fact.
The fallback may not change membership, normalize attention per chunk, reduce
width, or alter graph edges. A capacity failure blocks learning and reopens this
spec rather than silently shrinking the model.

## Acceptance Criteria

Local implementation is accepted only if:

1. One source module implements exactly the staged CNN, two sparse local
   blocks, one 17-query patch summary, one CLS-only global-token block, final
   norm, and FP32 five-logit head.
2. No local or global code path materializes patch-by-patch `N x N` attention.
3. Graph construction derives from `(x,y)`, handles holes/borders and includes
   self. Local softmax has exactly one valid null slot and no cardinality bias.
4. Every attention module has six heads and logically separate bias-free Q/K/V
   projections with Spec 0032's exact packed slice order and independent
   initialization. The global patch summary retains zero-initialized head
   offsets and fixed negative-log key-count calibration. The final CLS row uses
   softmax with no mask, cardinality bias, null or head offset. Local attention
   uses fixed-26 FP16 efficient SDPA in query chunks of 2,048; no alternate
   activation or backend is implemented.
5. Local blocks have no dynamic register; a selected static zero-value null key
   never receives patch state. Global tokens never update patches in the forward
   graph; there is exactly one patch-summary read.
6. Every attention and FFN is a separate pre-norm residual sublayer. Every FFN
   is the locked width-256 SwiGLU.
7. In an FP32 deterministic test, the optimized CLS-only output matches the CLS
   row of the full 17-token reference at `rtol=1e-5, atol=1e-6`. Define the
   reference loss from the CLS row only, then compare `T1` input gradients and
   every attention/norm/CLS-FFN parameter gradient at the same tolerances; the
   reference uses the same shared attention norm and unused output rows add no
   loss.
8. Tests inspect the global patch-summary core and require FP32 dot products,
   score/bias, sigmoid, V product and reduction. They separately require the
   final CLS row to use ordinary softmax attention without a mask or sigmoid
   offset. The large patch/local path remains FP16 under CUDA autocast; the
   calibrated global sigmoid core and final classifier/loss boundary remain
   FP32. Later CUDA tests inspect dtypes and prove the forced efficient backend.
9. Every trainable parameter, including the CNN, receives a non-`None`, finite
   gradient in a local backward test.
10. Normal/SO(2) classifier initial states and the single prebuilt graph object
   and identity hash are exactly matched; output label order matches dataset
   indices.
11. Optimizer-group tests apply `5e-3` decay to ordinary matrix weights while
   keeping relative-bias tables, semantic offsets, CLS/REG tokens, local null
   keys, biases, and normalization parameters out of weight decay.
12. Focused tests, full Ruff/BasedPyright, `git diff --check`, and repository
    preflight pass. Python changes additionally pass `./scripts/python_quality.sh`.
13. New model/graph source contains no Kaggle account, dataset or mount literal;
    the future execution-package owner/source contract above remains the only
    supported route for professor-account execution.

Before any learning package or run is proposed, separately authorized
largest-bag capacity evidence must report each representation independently with
finite forward loss, every gradient finite, committed optimizer steps, GradScaler
behavior and recorded VRAM headroom. One run's failure must not roll back or
invalidate a successful step by the other. That evidence is not
sufficient-learning evidence.

## Tests And Verification Commands

Expected focused test families:

```text
test_spec0026_graph.py
test_spec0026_attention.py
test_spec0026_mil_model.py
test_spec0026_capacity_package.py       # later execution spec owns its contract
```

Every test docstring states the invariant and why its regression would invalidate
the real complete-bag comparison. Required repository commands after Python
implementation:

```bash
.venv/bin/python -m pytest -q tests/test_spec0026_graph.py \
  tests/test_spec0026_attention.py \
  tests/test_spec0026_mil_model.py
./scripts/python_quality.sh
./scripts/agent_preflight.sh
git diff --check
```

The original implementation acceptance evidence is superseded by Spec 0032's
AMP/backend revision. Current verification status lives in `CURRENT.md`; the
accepted implementation remains `src/eqvae/models/local_global_mil.py`, publicly
exported through `src/eqvae/models/__init__.py`.

## Implementation Blockers

Local model/test implementation is complete. Capacity packaging is not
specified enough to implement yet. Remote capacity packaging, Kaggle
writes, classifier learning, and sealed-test evaluation each require separate
explicit user authorization. Learning remains blocked until the largest-bag
capacity/numerical gate passes and the full-coverage reader is integrated into
a development-only package.

## Known Risks

- Only 106 labelled training WSIs exist; the approximately 1.51M-parameter
  classifier can overfit despite sparse attention and early stopping.
- Attention-kernel evidence is not ovarian-MIL evidence. Global key-count bias
  is a stable prior, not proof of superior aggregation.
- Sixteen free REGs have no guaranteed specialization and can collapse to
  redundant queries.
- Omitting global coordinates intentionally limits acquisition/layout shortcuts
  but may also discard diagnostically useful macro-distribution information.
- A radius-2 lattice graph captures only connected local context after two
  layers; tissue holes and sparse foreground can disconnect regions.
- Fixed-degree gather/SDPA may dominate local softmax even though its memory is
  linear; Specs 0027–0028 measure only one local module.
- The larger width and complete patch CNN can still exceed VRAM without any
  quadratic attention matrix.
- The coordinate graph exposes degree, connectedness, and bag-cardinality
  structure even without positional token values; the model can exploit WSI
  geometry unrelated to morphology.
- One trajectory per representation cannot establish optimization-seed
  robustness or a general representation ranking.

## Adversarial Checks

- Mutate one coordinate while preserving row order and require graph identity
  or lattice validation to fail.
- Remove a border neighbour and verify degree/bias changes without padding mass.
- Duplicate or remove the selected local null slot and require its algebra test
  to fail.
- Add Q/K bias or alter the selected activation/normalization and require a
  structural test to fail.
- Change packed logical slice order/initialization or decay a relative-bias
  table and require module/optimizer contract tests to fail.
- Introduce a patch query over REG/CLS and require the forward-direction test to
  fail.
- Compute all 17 final rows or drop the CLS FFN and require equivalence/topology
  tests to fail.
- Share one LayerNorm between patch and global streams in cross-attention and
  require module/config identity checks to fail; retain the deliberately shared
  final self-attention norm required for exact CLS-row equivalence.
- Reorder diagnosis logits or let one VAE run derive a different graph and
  require comparison-contract tests to fail.
- Attempt chunk-local softmax normalization or patch subsampling and require the
  complete-bag capacity/package contract to fail.

## Downstream Evidence

Measured full-model capacity, throughput, residual/gate RMS, REG diversity,
attention maps, and development learning are
diagnostics produced after implementation; they do not reopen architecture
automatically. Any architecture change requires an explicit Spec 0026 revision
before code or execution.

## Evidence Basis

- Sigmoid self-attention with key-count calibration: Ramapuram et al.,
  [Theory, Analysis, and Best Practices for Sigmoid Self-Attention](https://arxiv.org/abs/2409.04431).
- The user-proposed domain application is useful motivation, not architecture
  validation: [arXiv:2604.27124](https://arxiv.org/abs/2604.27124).

## Related Requirements And Evidence

- Data and pairing: Specs 0018, 0020, 0023, and 0025.
- Previous capacity evidence: Spec 0023 and
  `runs/local/wsi45630_capacity/bundle/transformer_probe.py`.
- Attention pooling: https://arxiv.org/abs/1802.04712 and
  https://arxiv.org/abs/1810.00825.
- Canonical pre-norm ViT block: https://arxiv.org/abs/2010.11929.
- Sigmoid key-count bias and non-competitive attention:
  https://arxiv.org/abs/2604.27124.
- Local/neighbourhood MIL context:
  https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2024.1389396/full.
- Ovarian subtype context: https://pmc.ncbi.nlm.nih.gov/articles/PMC8070731/.

## Related Files

- `CURRENT.md`
- `GOAL.md`
- `docs/repo_goal_and_requirements.md`
- `docs/equivariant_vae_transition_plan.md`
- `docs/specs/README.md`
- `docs/specs/0023-matched-supervised-latent-evaluation.md`
- `docs/specs/0025-full-foreground-logical-mil-dataset.md`
- `src/eqvae/data/full_foreground_latents.py`
