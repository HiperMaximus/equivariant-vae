# Spec 0023: Matched Supervised Latent Evaluation

Status: tissue is frozen after passing at `1.5e-3`; corrected class-specific
MIL completed five epochs but still shows insufficient learning. The user
authorized the full-coverage real-WSI transformer capacity probe below;
classifier learning remains separately gated.
Implementation readiness: manifests, readers, classifiers, paired atomic
updates, exact schedules, one-off sweep/confirmation packages, focused tests,
and unchunked dual-T4 capacity are locally verified; both confirmation versions
and the epoch-5 horizon output are downloaded, and every boundary is hash
verified
Owner/workstream: downstream supervised evaluation
Last updated: 2026-08-30

## Purpose

Run two simple, controlled supervised experiments that test whether the frozen
continuous-`SO(2)` VAE latents are more useful than the frozen normal-VAE
latents:

1. five-class WSI ovarian-cancer diagnosis with attention MIL;
2. three-class high-purity tumor/stroma/necrosis patch classification,
   including a nested label-efficiency curve.

Both branches use the same examples, logical order, classifier initialization,
training order, validation policy, and metrics. The representation store is the
only intended experimental difference.

## Non-Goals

- No VAE fine-tuning, decoder use, sampled `z`, DINO stage, graph
  neural network, multiscale WSI reader, learned thumbnail proposal, or raw-RGB
  classifier.
- No patch-level cancer labels or treatment of WSI patches as independent
  cancer outcomes.
- No exhaustive segmentation claim from incomplete supplemental masks.
- No architecture or hyperparameter search per representation. A model-specific
  search would confound representation quality with tuning effort.
- No automatic architecture campaign. The completed horizon, width-128,
  class-specific, and scale-correction diagnostics did not establish sufficient
  MIL learning. The transformer capacity check is not a learning experiment;
  integral/sigmoid attention and equivariant convolutions remain deferred.
- No general training framework, multi-node support, CPU fallback, arbitrary
  latent shape, production deployment, or broad compatibility work. The code
  is for one paired experiment on one Kaggle dual-T4 setup.
- No paper conclusion, issue closure, or thesis update before the audited test
  outputs exist and are interpreted with the small-WSI caveats.

## Authorized Synthetic Transformer Capacity Probe

Locked / implementation-ready, 2026-08-29. The user explicitly requested a
direct synthetic fit check without loading datasets or running the full test
suite. This narrow probe changes no shared model, reader, trainer, or schedule.

- One standalone private Kaggle script; all input source lists empty. Generate
  one FP32 `[8149,16,32,32]` random bag in memory and copy it to each T4. The
  shape comes from the already-established largest training WSI, not a new
  manifest or binary read. These are synthetic replicas, not measured normal
  or SO(2) latent distributions.
- Copy the current three-stage Conv/GroupNorm/GELU patch encoder shape into
  this self-contained script. Encode every patch together, without chunking,
  detaching, sampling, or caching embeddings outside autograd.
- Width 128, four heads of width 32, one learned CLS, eight learned registers,
  and two full noncausal self-attention blocks: sequence length `8158`.
  Each block is pre-LayerNorm attention plus residual, then pre-LayerNorm
  SwiGLU (inner width 256) plus residual. No dropout, position encoding, causal
  mask, separate pooling, or attention-matrix output. Final normalized CLS
  feeds an FP32 `Linear(128,5)` and FP32 cross-entropy from raw logits.
- Use PyTorch SDPA's memory-efficient backend explicitly on the T4s; no silent
  math fallback or claim of FlashAttention. Record its successful execution,
  installed Torch/CUDA/cuDNN versions, device names/capabilities, and precision.
  Unsupported-backend failure is not proof that the architecture cannot fit.
- Both independent replicas have identical initial parameters and synthetic
  inputs. Convolutions use the existing Kaiming-normal rule, linear weights
  Xavier-uniform, norm scales one, all biases zero; CLS/register parameters
  use small normal initialization (std `0.02`). AdamW uses LR `2e-4`, weight
  decay `1e-4` for parameters with `ndim >= 2`, zero for other parameters.
  FP16 autocast and GradScalers `init_scale=32768`, `growth_interval=1000000`.
- Run one warmup and one measured full forward/backward/optimizer step per
  replica. Check both gradient sets before either optimizer step; no silent
  scaler skips or numerical retries. Include AdamW states and the complete CNN
  in peak allocated/reserved memory and synchronized paired wall timing.
  Publish a small JSON even on OOM or numerical/backend failure.
- This is capacity evidence only, not learning, real-input numerical stability,
  an epoch, a checkpoint, or authorization for training. No resumability or
  campaign framework is needed. Check syntax, metadata/source size, and one
  small CPU architecture smoke; use a focused independent review. The user's
  explicit scope excludes the full repository test suite for this probe.

Execution: private kernel
`maximusshtefan/eqvae-mil-transformer-synthetic-capacity` version 1 accepted on
2026-08-29; completed successfully. Source SHA-256
`f7cc5c0eea786efd030c4f80a87ab29b706d1a3195df7e0b3509d2d7e0f4e67b`.
The standalone tiny CPU backward, syntax, focused Ruff, source/metadata checks,
shell syntax, diff check, and repo preflight pass; independent clean-context
Terra review found no blocker. No full suite ran, per explicit user scope.
Downloaded result at
`runs/kaggle/ubc_ocean_mil_transformer_capacity_v1/transformer_capacity.json`,
SHA-256 `52343f00c19f5a6c38908c29b9f2c13b241b9c08da3613529abaee67c5e45239`,
reports `fits_synthetic_full_training_step`. Both replicas finish both steps
with finite gradients, scale 32768 unchanged, and 42 initialized optimizer
parameter states. Maximum warmup allocated/reserved memory is 2.965/3.756 GB
per T4; measured allocated memory is 2.100/2.101 GB, against 15.636 GB each.
The measured paired step is 0.13537 s, not a sustained throughput benchmark.
Torch `2.13.0+cu130`, CUDA 13.0, cuDNN 92000 execute the forced memory-efficient
SDPA backend successfully on SM75 T4s. No chunking or data attachment was used.
Capacity passes; actual-latent learning remains untested for this architecture
and requires a separately authorized diagnostic.

## Authorized Full-Coverage WSI45630 Capacity Probe

Locked / implementation-ready, 2026-08-30. Run one private dual-T4 memory
check on the actual frozen embeddings of all **32,595** full-grid Otsu patches
of training WSI45630. The user requests this follow-up to the completed
single-WSI extraction, not a learning run or expanded campaign dataset.

- Reuse 5,136 foreground rows in base part 4 and 4,810 rows in part 11;
  supplement with all 22,649 rows from the completed
  `maximusshtefan/eqvae-wsi45630-completion` version 1. Assign the supplement
  part 12 only inside this probe's private catalog. Existing learning manifests
  and catalog stay unchanged; no `.bin` is copied, regenerated or downloaded.
- Derive exact `(atlas_row_index,wsi_id,x,y,part,file_index)` pointers from
  the hash-bound candidate, part-4 work, part-11 and completion manifests.
  Filter WSI45630 before interpreting mixed metadata rows, but enumerate the
  original physical file before filtering. Require exact target coverage,
  disjoint sources, unique coordinates and sorted `y,x` order for both models.
  Exclude the 57 mask-only atlas rows. No validation/test logical files,
  labels, or payload rows are read.
- Stage a small private dataset `maximusshtefan/eqvae-wsi45630-capacity-inputs`
  with only this bag's pointers/catalog, the existing `eqvae` source snapshot
  (preserving the reader and paired-init package dependencies), the
  previously tested standalone transformer model source, and their hashes.
  Byte-verify private version 1 before launching
  `maximusshtefan/eqvae-wsi45630-transformer-capacity`. Attach only that dataset
  and the three relevant latent-producing kernels, never the raw competition
  or a sealed logical test dataset. Physical mixed shards remain read-only;
  only the fixed training pointers may be consumed.
- Resolve the part-11/completion filename collision using their distinct
  authenticated sidecar hashes. Reuse `SupervisedLatentStore` unchanged for
  header/sidecar validation and complete, logically ordered reads; no full
  binary rehash or GPU-side bag truncation. Record actual row/part counts and
  shared pointer identity. Upload no source archive or manifest inside `run.py`.
- Reuse `make_model` from the successful synthetic probe without changing
  architecture or initialization: complete CNN, two pre-LN residual full-SDPA
  blocks, width128/four heads, SwiGLU256, one CLS, eight registers, FP32 head.
  Sequence length is **32,604**. Both independent classifiers begin identical;
  normal uses GPU0 and SO(2) GPU1. Keep FP32 stored inputs/parameters, FP16
  autocast, explicit `EFFICIENT_ATTENTION` only, AdamW `2e-4`, matrix-only
  decay `1e-4`, and GradScalers `32768` / `1000000`.
- As in the synthetic capacity control, use unweighted FP32 cross-entropy
  for the fixed EC target (index 1). Execute one warmup and one measured
  forward/backward/AdamW step per branch; check both complete gradient sets
  before either optimizer step. No retries, fallback, chunking, detached CNN,
  pooling changes, epochs, validation, learning gate, or resumable checkpoints.
- Publish only compact JSON/log evidence: actual topology and input hashes,
  read counts, shape, runtime/backend, finite losses/gradients, scaler values,
  all optimizer states, paired wall time, peak allocated/reserved memory,
  and explicit OOM/numerical/backend failure even when the probe fails.
  Success establishes two-step full-bag fit, not convergence or stability over
  training. Run focused mapping/mount/architecture checks and independent
  clean-context review. Preserve the user's small capacity-probe testing scope;
  run the repository quality gate with only these focused pytest cases selected,
  retaining repository-wide lint/type checks rather than repeating all tests.

## Inputs And Frozen Data Contract

Local implementation status: `build_ubc_supervised_manifests.py` has emitted
and audited the 12-row catalog, six WSI CSVs, and seven tissue CSVs under
`runs/local/ubc_ocean_supervised_manifests` without reading any latent binary.
The narrow local preflight now resolves synthetic mounted parts through that
catalog contract, restores grouped physical reads to logical bag order, and
proves the paired models and nested epoch order without touching real payloads.
The one-off calibration implementation now lives in
`supervised_calibration.py`, `build_ubc_supervised_calibration.py`, and the
single Kaggle `run_template.py`. Kaggle limits one script source to less than
1 MB, so the wrapper contains executable orchestration and its compact config
only. Following the established Spec 0021 resume-input pattern, the sweep uses
one immutable private input dataset containing the exact source tree, catalog,
106-WSI training manifests, and 5,671-per-class tissue training manifest. Its
versioned receipt binds the complete canonical bytes before the kernel may be
built or pushed. The sweep input contains no validation or test artifact. The
later confirmation uses a separately sealed learning-input dataset that adds
validation and the human audit binding the sweep artifact, fixed EWMA rule,
rationale, and one shared peak per task; it still contains no test artifact.
Neither input dataset contains a latent binary or a physical split copy.

- Completed Spec 0021 base latent-store global audit.
- Completed Spec 0022 top-up pair and WSI logical-dataset audit.
- Normal and SO(2) posterior `mu` records are read-only FP32 `(16,32,32)` and
  aligned by the same model-independent logical CSVs.
- No latent binary is downloaded, copied, concatenated, or rewritten. One
  compact physical-parts catalog resolves `(part,model_name)` directly to the
  read-only Kaggle mount, filename, row count, bytes, and SHA-256. Parts 1-5
  are the five completed base pairs and part 11 is the completed additive pair;
  part 11 has no special loader behavior.
- Frozen WSI split: 106 train, 23 validation, 23 sealed test, with diagnosis
  quotas from Spec 0018. A WSI may appear in exactly one split across both
  tasks.
- WSI task consumes three unified instance CSVs and three bag CSVs derived from
  the completed Spec 0022 files. Final instance rows contain
  `part,file_index`, never a base/top-up discriminator. Each bag contains the
  complete fixed `K_w` selection; there is no per-epoch patch subsampling.
- Tissue task starts from the existing Spec 0019 tissue views. Training uses a
  fixed balanced pool of exactly 5,671 tumor, 5,671 stroma, and all 5,671
  necrosis rows. Validation and test remain the complete natural-distribution
  views: 31,339 and 31,572 rows respectively.
- Tissue train membership uses one deterministic, nested, WSI-stratified
  priority order per class. Within every `(class,wsi_id)` group, permute rows
  once with
  `PCG64(SeedSequence([20260827,tissue_class_index,wsi_id]))`. Put the first
  row from every positive WSI first; order all remaining rows by increasing
  normalized within-WSI rank `rank / available_in_that_WSI`, breaking ties by
  `wsi_id` and within-WSI rank. This guarantees every eligible WSI appears in
  the smallest subset while later prefixes grow approximately in proportion
  to per-WSI availability. Necrosis eventually uses all rows. Tumor and stroma
  stop after 5,671. Selection is without replacement and no row is resampled
  during training.
- Nested tissue train CSVs contain exactly `[250,500,1000,2500,5671]` rows per
  class. Each is the prefix of the same per-class priority order, so every
  smaller subset is a strict subset of every larger one.

Physical catalog file:

```text
physical_parts.csv
```

Each catalog row contains:

```text
part,model_name,kaggle_source,binary_name,row_count,binary_bytes,binary_sha256,sidecar_name,sidecar_bytes,sidecar_sha256
```

The catalog key is `(part,model_name)`. Both model rows for one part have the
same record count and coordinate order. A logical row selects only
`part,file_index`; choosing `normal_vae` or `so2_vae` resolves the corresponding
catalog row without changing logical membership.

WSI logical files:

```text
wsi_cancer_train_instances.csv
wsi_cancer_train_bags.csv
wsi_cancer_validation_instances.csv
wsi_cancer_validation_bags.csv
wsi_cancer_test_instances.csv
wsi_cancer_test_bags.csv
```

Each WSI instance row contains:

```text
instance_row,atlas_row_index,wsi_id,x,y,diagnosis_label,diagnosis_index,split,part,file_index
```

Each bag row contains:

```text
bag_row,wsi_id,diagnosis_label,diagnosis_index,split,instance_start,instance_count
```

The bag range addresses contiguous rows in its logical instance CSV. Physical
reads are grouped by part and increasing `file_index`; if read order differs
from logical order, the loader restores canonical `wsi_id,y,x` order before
returning the complete bag.

Tissue logical files:

```text
tissue_train_0250_per_class.csv
tissue_train_0500_per_class.csv
tissue_train_1000_per_class.csv
tissue_train_2500_per_class.csv
tissue_train_5671_per_class.csv
tissue_validation.csv
tissue_test.csv
```

Each tissue row contains:

```text
dataset_row,atlas_row_index,wsi_id,x,y,tissue_label,split,selection_rank,part,file_index
```

All logical CSV bytes and SHA-256 values are recorded before any classifier is
trained. The same files are used for both latent models.

For every nested CSV, the audit records rows per WSI and `n_WSI` per class and
requires every WSI positive for that class to be represented. Selected rows are
restored to authoritative `wsi_id,y,x` order; `selection_rank` retains the
nested-prefix identity.

## Leakage And Stable-Membership Contract

- Dataset membership is fixed in CSVs. Loaders may never select a fresh random
  subset, sample with replacement, or silently drop a WSI/patch because of its
  class, loss, attention score, or model representation.
- Ordinary training-order shuffling is allowed and recommended. For each seed,
  both representation branches receive the same shuffled WSI or patch indices
  in the same step. Shuffling changes order only, never membership.
- WSI train/validation/test disjointness is audited from `wsi_id`, not inferred
  from physical shards. Physical `.bin` files deliberately contain mixed
  splits.
- Training code receives only train and validation logical indices. The test
  loader is not constructed and test locations are not read until the complete
  protocol, training configs, selected validation checkpoints, and artifact
  hashes are frozen.
- Every run writes an access transcript of logical and physical row identities.
  Before test release, the audit proves the train/validation transcript has
  empty intersection with both tasks' test locations.
- Test predictions are made only after the paired normal/SO(2) development
  results are complete. No test-driven restart, epoch selection, threshold,
  architecture change, or subset change is allowed.

## Shared Spatial Patch Encoder

Both tasks use the same modest trainable spatial compressor over each frozen
latent map:

```text
FP32[16,32,32]
Conv2d(16,32,kernel=5,stride=2,padding=2) + GroupNorm(8 groups) + GELU
Conv2d(32,64,kernel=3,stride=2,padding=1) + GroupNorm(8 groups) + GELU
Conv2d(64,128,kernel=3,stride=2,padding=1) + GroupNorm(8 groups) + GELU
AdaptiveAvgPool2d(1)
flatten -> 128-dimensional patch vector
```

Convolutions perform the requested learned spatial downsampling. The 5-by-5
first layer supplies a broader early receptive field while the later 3-by-3
layers keep the model small. GroupNorm avoids dependence on variable MIL bag
or chunk batch statistics.

No rotation or spatial augmentation is applied to latent maps. A spatial image
rotation is not, by itself, the correct action on the SO(2) latent field
channels and would make the two branches incomparable.

## Experiment 1: WSI Attention MIL

- The patch encoder maps every instance in a WSI bag to a 128-vector.
- Primary aggregator: simple gated attention MIL:

  ```text
  score_i = w^T(tanh(V h_i) * sigmoid(U h_i))
  alpha = softmax(score over every patch in the WSI)
  wsi_vector = sum_i(alpha_i * h_i)
  logits = Linear(128,5)(wsi_vector)
  ```

  The attention hidden size is 64. Attention is permutation-invariant and is
  normalized over the complete fixed bag, not a sampled patch subset.
- Deferred optional control: if time remains after the six-run primary campaign,
  train one unweighted mean-pooling WSI run with the same patch encoder shape
  and protocol. It is not part of the initial experiment budget.
- Loss: diagnosis-frequency-weighted five-class cross entropy computed from
  the 106 training WSIs only.
- One optimizer step consumes one complete WSI bag. WSI order is shuffled in a
  paired manner. The capacity preflight first attempts the complete patch tensor
  through the CNN without chunking because this is the simplest exact execution.
  If and only if that fails for CUDA memory, use one locked activation-
  checkpointed chunk size: each chunk passes through
  `torch.utils.checkpoint(..., use_reentrant=False)`, all resulting token tensors
  remain attached to the autograd graph, tokens are concatenated, and one global
  attention softmax and loss cover all `K_w` instances. Detaching/caching tokens,
  freezing the patch encoder, or normalizing attention within chunks is
  forbidden.
- Export per-WSI predictions and coordinate-linked attention ranks. Attention
  is interpretability evidence, not proof of causal or histologic importance.

## Experiment 2: Tissue Classification And Label Efficiency

- Apply `Linear(128,3)` to the shared patch encoder output.
- Use ordinary unweighted three-class cross entropy because every training CSV
  is exactly class-balanced.
- Train independently on each of the five frozen nested training CSVs.
- Validation and test are never balanced or resampled. They retain their fixed
  acquired distributions and WSI identities.
- Patch batches are shuffled without replacement inside the fixed CSV. Both
  representation branches receive identical logical batch indices.

## Paired Training And Hardware Contract

- Exactly one Kaggle dual-T4 setup is supported. Use a single controller, not
  DDP: normal-latent classifier on `cuda:0`, SO(2)-latent classifier on
  `cuda:1`.
- At the start of every paired run, create one CPU classifier state and load
  byte-identical copies into both branches. Use the same optimizer family,
  learning rate, weight decay, epoch limit, mixed-precision policy, batch/order
  indices, and validation cadence.
- Initialize convolution weights with Kaiming normal initialization using
  `mode="fan_out"` and the standard ReLU gain as the fixed GELU-compatible
  approximation,
  initialize every attention and classifier `Linear` weight with Xavier uniform
  gain `1.0`, initialize GroupNorm scales to one, and initialize every classifier
  bias to zero: every `Conv2d`, every `Linear` including `V/U/w` and the final
  heads, and every affine `GroupNorm` has zero bias. In particular, the final
  classification heads are not
  zero-weight initialized: both branches must propagate gradients through the
  complete downstream network on their first update.
- The initial campaign contains six paired runs with one initialization per
  configuration:

  | Configuration | Initialization/order seed |
  | --- | ---: |
  | WSI attention MIL | 1701 |
  | tissue 250/class | 3407 |
  | tissue 500/class | 3407 |
  | tissue 1,000/class | 3407 |
  | tissue 2,500/class | 3407 |
  | tissue 5,671/class | 3407 |

  The normal and SO(2) branches share the exact initialization state and
  shuffled index order within each row. Dataset membership remains governed by
  the separate fixed selection seed `20260827`; classifier seeds may never
  regenerate a logical CSV.
- All five tissue sizes use the same initial state. For epoch `e`, derive one
  permutation from the full 5,671-per-class nested training pool using
  `SeedSequence([3407,e])`, then filter that order to the current subset. Thus a
  smaller subset's epoch order is a subsequence of the larger subsets' order. A
  subset starts the next deterministic epoch whenever its filtered order is
  exhausted. Training uses `drop_last=True`: discard that epoch's short tail
  before deriving the next deterministic permutation. It never samples outside
  its fixed CSV or samples with replacement inside one epoch.
- MIL uses the corresponding complete-bag permutation from
  `PCG64(SeedSequence([1701,e]))`. Sweep, confirmation, and the later campaign
  each reconstruct fresh models, optimizers, scalers, and schedule state from
  the task seed; no learned or optimizer state transfers between phases.
- All five tissue sizes receive the same maximum of 30 epochs, full validation
  every half epoch, and patience of 5 epochs (10 half-epoch validation checks).
  For a tissue configuration, one epoch means one traversal of that
  configuration's fixed subset with its short final batch dropped; it is not a
  partial traversal of the 5,671-per-class pool. This gives every available
  labelled row approximately the same maximum number of training exposures.
  Larger subsets naturally contain more optimizer updates per epoch, so the
  result is explicitly a fixed-epoch label-efficiency curve, not an equal-
  compute comparison.
- Every size still has one optimization trajectory, so the initial
  label-efficiency curve cannot estimate training-seed variance. If time
  permits, repeat the whole six-run campaign with a second seed schedule chosen
  before inspecting test results.
- Exact bitwise CUDA determinism is not required, but paired initialization,
  order, and data identity are recorded.
- The development campaign contains exactly six paired runs: one WSI attention
  run plus five tissue subset-size runs. Each
  completed paired run is an atomic campaign boundary. Reuse the proven VAE
  durability and evaluation pattern at two committed boundaries per epoch:
  `epoch + 0.5` and `epoch + 1.0`. Each boundary runs the complete validation
  split for both branches, updates their best-checkpoint and patience state, and
  writes a full resumable paired checkpoint. Internally, the half boundary is
  after `floor(S/2)` successful paired optimizer steps and the epoch end after
  all `S` steps, where `S` is 106 complete WSI bags for MIL and
  `floor(n_train_patch/128)` for that tissue subset. For odd `S`, the first half
  has `floor(S/2)` steps and the second has the remainder; external progress and
  checkpoint names remain expressed in completed epoch fractions rather than
  absolute batch numbers.
  Each atomic checkpoint stores both model states, both optimizers, both
  GradScalers, completed epoch and within-epoch cursor, deterministic paired
  order identity, best-validation state and patience counters, histories,
  access-transcript prefix, and campaign progress. Write the checkpoint and
  adjacent progress artifacts through temporary files before replacement, hash
  them, and expose the boundary as resumable only after the complete paired
  bundle is committed. A wall-clock guard may stop only at such a boundary and
  must leave enough time to publish the resumable prefix inside Kaggle's 8-hour
  session limit.
- Early stopping is paired. Each branch owns its validation macro-F1/loss best
  checkpoint and patience counter, but both continue through the same epoch
  fractions until both have simultaneously exhausted their task's patience or
  the shared 30-epoch ceiling is reached. A branch that exhausts patience first
  is not frozen or removed; a later improvement resets its own counter. This
  keeps exposure and order matched while allowing each representation to select
  its own best validation checkpoint.
- Before the first real run, seal an input-mount manifest binding every row of
  the physical-parts catalog, the compact logical datasets, checkpoint hashes,
  exact mounted byte totals, and paths. Input mounts stay read-only; no latent
  is copied into `/kaggle/working`.
- A dual-T4 preflight must measure the maximum 8,149-instance attention bag
  forward/backward peak memory and settled step time. It first runs the exact
  unchunked complete-bag step on both branches. A successful unchunked result
  locks `checkpoint_chunk_size = null` for training. Only a CUDA-memory failure
  authorizes a second probe with activation checkpointing and a smaller locked
  chunk size. This proves that one complete WSI optimizer step fits safely on
  each T4. Tissue uses the ordinary fixed batch size of 128 and needs no separate
  capacity or throughput probe.
- The one-off probe is train-only WSI `45630`: 3,339 pointers in part 4 and
  4,810 pointers in part 11. It mounts only those two completed Kaggle kernel
  outputs, performs one warmup and one measured FP16-autocast/GradScaler AdamW
  step per branch, and writes only `spec0023_mil_capacity_probe.json` with a
  100,000-byte projection under Kaggle's 20 GB output limit. The artifact
  records the 8-hour session limit, one-hour required reserve, elapsed and
  remaining time, and that the test split was neither authorized nor mounted.
  Per the accepted compact-audit policy, the probe does not reread 25.24 GB to
  recompute binary hashes: completed audit/catalog SHA-256 values bind the
  sources, while the mounted reader rechecks exact file sizes and header row
  counts before reading the indexed records.
- Before the six-run campaign, run exactly two small paired learning-rate range
  phases. MIL uses one complete 106-WSI training epoch. Tissue uses one complete
  132-batch epoch of the fixed 5,671-per-class subset; the selected tissue peak
  is then reused unchanged by all five tissue sizes. Each phase starts normal
  and SO(2) classifiers from the same task seed and increases LR exponentially
  from `1e-5` through `3e-3` over its successful paired updates. For
  `k=1..S`, use
  `lr(k)=1e-5*(3e-3/1e-5)**((k-1)/(S-1))`, where `S=106` for MIL and `S=132`
  for tissue, and record the LR before each paired attempt. Both branches
  receive the same logical order and LR. A skip, nonfinite value, or divergence
  stops both together and records the last aligned prefix. The range phases use
  training labels only; they do not mount, construct, or access the test split.
- A paired update is atomic across representations: run both forwards and
  backwards, unscale both optimizers, and check both losses and every gradient
  before either optimizer steps. Only when both sides are finite may both
  optimizers step and the single successful-pair/LR cursor advance. A failure
  commits neither side and retains only the preceding aligned prefix.
- Sweep version 1 committed 11 paired MIL updates, then failed atomically on
  train-only WSI `65094`, the first MC bag in the locked seed-1701 order. Its
  finite class-weighted loss produced a nonfinite gradient before either AdamW
  step. This bag is complete and unchunked: 5,773 instances. MC has 8 of the
  106 training WSIs, so its fixed inverse-frequency weight is `2.65`.
  Multiplying that weight into the scalar loss before GradScaler backward makes
  the effective FP16 scale `32,768 * 2.65 = 86,835.2`, above FP16's largest
  finite value, whereas the earlier EC capacity bag used weight `0.8833...`.
  The failure is therefore a numerical-execution defect, not evidence that the
  bag does not fit or that LR `1.8176402909092286e-5` is too large.
- The one-off replacement package must first reproduce the version-1 failure
  from a fresh seed-1701 state using the same first 12 complete training bags,
  order, exponential LRs, optimizer settings, and scale `32,768`. Failure
  evidence records the branch and first affected named parameter, gradient
  dtype and nonfinite counts, current scale and class weight, plus finite/NaN/
  Inf summaries for the complete input bag. A second fresh replay applies the
  correction below and must commit the same first 12 paired attempts. The
  replacement sweep then starts fresh; neither replay state is reused.
  Mechanically anchor the replay to version 1: sweep-audit SHA-256
  `654c8c244698717c1a628e3b395c9b414f9e1945cafc1fdd893ce1457d6246ef`,
  MIL initialization SHA-256
  `ae60948164a694c65c6b758e556d7199ccdf325fd5b35ea4fb6c3f7a3b815a1d`,
  and order SHA-256
  `8267113c14c486039933a1fb15a3ae35fb2e608ae6abd5ac9e758332a69940cb`.
- For weighted batch-one MIL only, compute and report `weight_y * CE`, but
  backpropagate `CE` through the FP16-autocast graph. After each GradScaler has
  unscaled its optimizer's FP32 parameter gradients, multiply every gradient
  in that branch by the same `weight_y`, then perform the existing paired
  all-gradient finite check before either optimizer steps. This is exactly the
  same weighted gradient as backpropagating `weight_y * CE`; it only moves the
  scalar multiplication past the FP16 activation-gradient path. Both
  representations receive the same label-derived weight. Tissue remains
  unweighted. Keep initial scale `32,768`, growth interval `1,000,000`, complete
  bags, and `checkpoint_chunk_size = null` unchanged. If the corrected replay
  still has a numerical failure, stop as blocked. A later numerical boundary in
  the single fresh replacement sweep follows the common selection rule below:
  after at least ten aligned updates it exposes the preceding committed LR as
  the upper bound and remains selectable only when both EWMAs already contain a
  common clearly descending region. Do not lower the scale again, chunk a bag,
  or open a broader search.
- The complete version-1 artifact remains immutable failure evidence. The
  replacement package reruns both MIL and tissue from fresh task seeds and is
  the single authoritative sweep for human peak selection; curves may not be
  cherry-picked across versions. It uses a newly sealed train-only input
  dataset because the source and this spec changed. No validation or test
  logical artifact is mounted or constructed.
- The replacement audit records the resolved post-upgrade torch, torchvision,
  torchaudio, torch-CUDA/cuDNN versions, both T4 names and capabilities. Later
  confirmation accepts it only when its embedded sweep config matches the
  locally byte-verified v2 private-input receipt, source-tree/spec/manifest
  hashes, latent source records, output allow-list, complete recovery record,
  and sealed version-1 replay identities.
- Historical sweep evidence is validated against the immutable spec SHA-256 in
  its own byte-verified input contract. The later confirmation package binds the
  then-current spec separately. Updating live status text after a launch must
  not falsely claim that the new bytes were part of the historical sweep, and
  must not invalidate otherwise unchanged sealed evidence.
- Before either task opens an mmap, authenticate every mounted producer
  sidecar against the sealed catalog's exact name, byte count, and SHA-256;
  require its complete schema/model/tensor/file-size/payload-CRC contract to
  match the mounted binary header and catalog. Together with Kaggle's immutable
  mounted kernel output, this preserves the producer's prior full-byte audit
  without rereading roughly 73 GiB for every one-off calibration run.
- This is one compact, experiment-specific calibration, not a reusable tuner or
  exhaustive search. Implement it as the smallest direct one-off path over the
  existing readers and classifiers. Do not add candidate grids, repeated seeds,
  per-subset searches, Bayesian/automated optimization, generalized campaign
  infrastructure, or extra trials to improve a merely adequate finite result.
  One paired sweep package may run the MIL and tissue phases sequentially; after
  the two peaks are sealed, one paired confirmation package may run both short
  confirmations sequentially.
- Select one shared peak per task from the common stable descending region of
  both branch curves, conservatively bounded by the branch that loses stability
  first. Do not select separate normal/SO(2) peaks and do not tune the five
  tissue sizes independently. Record the curves, smoothing rule, selected peak,
  and rationale in one compact audit before confirmation.
- Both curves record raw loss and an EWMA with `alpha=0.1`, initialized from the
  first raw loss. Numerical failure means a nonfinite loss or gradient, or any
  GradScaler skip. After ten aligned updates, divergence means either branch's
  EWMA exceeds four times its lowest earlier EWMA. The selected shared peak is a
  human-sealed conservative point from the clearly descending region of both
  EWMAs, no later than the last aligned LR before either branch's numerical
  failure or divergence; it is never auto-selected.
- Confirmation packaging accepts only a hash-bound sweep audit with at least ten
  committed aligned updates per task and internally consistent paired curves,
  attempts, successful-update counts, and exponential LR prefixes. A complete
  sweep exposes its final committed LR as the upper selection bound; a numerical
  failure exposes the preceding last committed LR; a divergence excludes the
  divergent committed point itself. An earlier failure is selection-blocked,
  and a human peak above either task's audited bound is rejected.
- Confirm each selected task peak from a fresh paired initialization for exactly
  two epochs. Let `S` be the task's successful updates per epoch and
  `W = ceil(0.10*S)`. The schedule is zero before training; successful warmup
  update `k=1..W` uses `peak*k/W`, so no optimizer update is wasted at exactly
  zero. Hold the peak for the remainder of epoch one and all of epoch two, and
  run the complete validation split plus a full paired checkpoint every half
  epoch. Confirmation is a pass/fail learning
  and stability gate, not another validation-driven search: both branches must
  remain aligned and finite with zero scaler skips and a decreasing smoothed
  training loss. A hard numerical failure permits exactly one fresh fallback at
  half the sealed peak for that task; insufficient learning without a numerical failure is a
  blocker, not permission to search upward, add trials, or inspect test data.
  Each branch passes the learning check only when its final EWMA is below its
  EWMA at the end of warmup. A fallback reconstructs fresh task state and the
  confirmation audit seals the resulting effective peak.
- Confirmation packages contain only train and validation logical CSVs. They do
  not embed or construct a test CSV/index and may not read any test
  `(part,file_index)`, even though the required read-only physical binaries mix
  splits. The audit records this logical-access distinction.
- The finite initial MIL confirmation at `2e-4` did not establish learning in
  two epochs. A subsequent explicit scientific decision authorizes exactly one
  fresh MIL-only confirmation at shared peak `2.5e-4`, the rounded upper edge
  of the already observed common descending sweep region (`2.4655e-4`). Keep
  the 10%-of-epoch warmup so LR is the only changed training parameter. Tissue
  remains frozen at its sealed `1.5e-3` pass and is not opened, trained, or
  validated in this retry. This is not permission for another sweep or any
  further LR candidate.
- The subsequent scientific horizon decision reuses the exact confirmation-v1
  `2e-4` paired epoch-2 boundary rather than starting a third LR trial. Resume
  from
  `mil/initial/epoch_2.0/{manifest.json,progress.json,paired_checkpoint.pt}`;
  the checkpoint SHA-256 is
  `d9488f01e466987fe0942b1ad29e70b58d56325efd1082a4cd946714318a21de`.
  The adjacent manifest and progress SHA-256 values are respectively
  `825ea820ede5ff530b5d7d04d1c7f68948e63cc167a262b56f7146dacc678f75`
  and
  `d2425f0f3c2f5fc1add68aa2a867686e7fbd7a44d4260300cf51264f89420072`.
  The confirmation-v1 config and audit SHA-256 values are respectively
  `c03a31aa817c95fc90cc232dda5f97736c46a492e4cf7b5f6ec3ee2eab6cfa42`
  and
  `ec7a3554e4864b328dd86485c30eef6d5a726fc014acf89012bca07aea529571`.
  Confirmation-v2 at `2.5e-4` is forbidden as a resume source.
  Authenticate that boundary against its manifest, progress file, downloaded
  confirmation-v1 audit/config, and the current train/validation logical-manifest
  hashes before restoring either branch. Publish these exact small resume bytes
  as a private Kaggle input dataset and attach it explicitly; do not embed the
  checkpoint in `run.py` and do not depend on an implicit old-kernel mount.
  Its verified receipt pins the private dataset reference/version, input-
  contract SHA-256, every remote file byte count/hash, current continuation
  source-tree/spec hashes, and the exact train/validation plus physical-catalog
  hashes. The kernel checks those bindings, `checkpoint_chunk_size = null`, the
  single AdamW parameter-group schema with weight decay `1e-4`, both scaler
  states, and peak `2e-4` before any mmap or checkpoint restore.
- Continue the restored classifiers, AdamW optimizers, GradScalers, RNG,
  validation/best/patience state, and paired campaign state from epoch `2.0`
  through epoch `5.0`. Hold the already-reached shared LR at `2e-4`; warmup is
  not repeated. Epochs 3-5 use the ordinary seed-1701 epoch permutations and
  one complete unchunked WSI per paired update. Validate and commit a complete
  resumable paired boundary at `2.5`, `3.0`, `3.5`, `4.0`, `4.5`, and `5.0`.
  Tissue remains unopened and frozen. The kernel publishes the finite evidence
  even when learning remains insufficient; it must not intentionally convert a
  scientific negative result into a Kaggle runtime error.
- Reconstruct the pre-resume access-prefix hash and row count from the frozen
  logical bag orders without rereading latent payloads, and require them to
  equal the checkpoint. Continue that same transcript through epoch 5. The
  output audit retains the complete pre/post-resume training and validation
  histories, both branch states, schedule identity, and all six new boundary
  hashes. Report, without auto-selecting a new hyperparameter, whether each
  branch's epoch-5 EWMA and epoch-5 mean weighted loss are below its epoch-2
  values and whether its best post-resume validation point improves on its
  pre-resume best. These are diagnostic observations, not permission to inspect
  test data or discard one branch.
- If the epoch-5 evidence does not justify retaining the current aggregator,
  later architecture diagnostics proceed only in this order and remain shared
  between representations: (1) widen the gated scorer from 64 to 128 while
  retaining one global attention map; (2) make the gated attention maps and
  row-wise WSI classifiers class-specific; (3) average multiple genuinely
  independent gated score signals before softmax as the lambda-free integral
  analogue. A branch set that differs only in its final linear score vector is
  not a valid integral implementation because that average collapses to one
  ordinary linear scorer. Exact run length and acceptance behavior for each
  architecture stage must be locked here after the preceding evidence is
  inspected and before its implementation or Kaggle launch.
- The first architecture diagnostic is now locked. Start both branches from a
  fresh paired seed-`1701` initialization and change only the gated-attention
  scorer width from `64` to `128`: `attention_v` and `attention_u` become
  `Linear(128,128)`, while `attention_w` remains `Linear(128,1)`. Keep the
  exact seed-1701 width-64 initial patch-encoder and head parameters; only the
  dimension-changing attention matrices receive new shapes/values. Keep the
  128-dimensional patch tokens, one global attention map, five-class linear
  head, complete unchunked bags, AdamW weight decay `1e-4`, FP16/GradScaler
  settings, class weights, data/order, and shared peak `2e-4` unchanged. Train
  for five fresh epochs with the same 11-update linear warmup in epoch one and
  hold the peak thereafter. Validate and write a fully resumable paired
  boundary at every half epoch from `0.5` through `5.0`. Tissue and test data
  remain absent. The exact width-64 reference is downloaded horizon audit
  SHA-256
  `2edeb5d20471ebc559e3bb7f6f7cde4aeaa06723b98bdcfc28a7dca4039525fb`.
  Report stability, epoch-wise mean/EWMA training loss, all validation points,
  and whether each branch's best validation macro-F1 exceeds its width-64 best
  (`0.24500` normal, `0.24644` SO(2)). A finite negative result is published
  successfully and inspected before the class-specific stage; it does not
  trigger an LR change, fallback, extra seed, or automatic architecture choice.
- A later convolution-stage question may compare GELU with the VAE's learned
  scalar gate `x * sigmoid(a*x+b)`, initialized at `a=1,b=0`. That is not a
  drop-in activation-only change: the VAE policy also gives gate parameters
  zero weight decay and an LR multiplier of `0.5`. It is therefore deferred
  until the attention stages have been evaluated and must be specified as its
  own paired architecture/optimizer change before use.
- Once a MIL architecture/horizon diagnostic establishes adequate learning,
  freeze one epoch-normalized schedule per task:
  the same zero-to-peak linear warmup over the first 10% of epoch one, then
  cosine decay without restarts to 1% of the peak at epoch 30. For a
  configuration with `S` updates per epoch, let `W=ceil(0.10*S)` and `N=30*S`.
  Warmup uses `peak*k/W` for `k=1..W`; for `k=W+1..N`, use
  `peak*(0.01 + 0.99*(1 + cos(pi*(k-W)/(N-W)))/2)`, which reaches exactly
  `0.01*peak` at `k=N`. Each tissue size derives its own `W` and `N` from its
  fixed-subset `S` while reusing the same sealed tissue peak. Advance it after every
  successful paired optimizer update using completed epoch fraction and save
  the one shared successful-pair counter and schedule parameters in every
  half-epoch checkpoint. No later LR search is
  allowed after the compact calibration audit is sealed.
- Fixed non-LR training defaults:
  - AdamW uses the VAE semantic grouping: trainable parameters with `ndim >= 2`
    receive the task's matrix-only decay, while all scalar/vector parameters
    (`ndim < 2`), including every bias and GroupNorm affine parameter, receive
    decay `0`. The future tissue campaign uses matrix-only `5e-3`, matching the
    selected MIL value. This user decision supersedes prior future-tissue
    `1e-4` notation only; already-sealed tissue calibration and runtime-probe
    artifacts retain their historical `1e-4` values and are not rerun. The task
    peak is supplied only by the sealed paired calibration above. The capacity
    probe's `3e-4` was a numerical-safety coordinate, not a frozen learning-
    rate result;
  - FP16 autocast with GradScaler initial scale `32,768`. The real largest-bag
    probe showed the default `65,536` scale skipped the normal branch's warmup,
    while its automatic `32,768` retry and the SO(2) branch were finite; starting
    both at `32,768` preserves the paired first update. Version 2 confirmed both
    the first and second full-bag steps were finite without a scaler skip when
    both branches started at `32,768`. Use growth interval `1,000,000`, beyond
    this campaign's possible update count, so automatic growth cannot return to
    the known-bad `65,536`; any later skip fails the paired run visibly;
  - tissue batch size `128`, `drop_last=True`, maximum `30` epochs, validation
    every half epoch, and patience `5` epochs / `10` validation checks;
  - MIL batch size one complete WSI, maximum `30` epochs;
  - MIL validation every half epoch and patience `12` epochs / `24` validation
    checks;
  - retain the checkpoint with highest validation macro-F1 for each branch and
    seed, breaking ties by lower validation loss then earlier epoch;
  - initial MIL execution is unchunked (`checkpoint_chunk_size = null`); only an
    observed CUDA-memory failure opens a checkpointed fallback, beginning at
    chunk size `128`; the successful exact mode is then fixed for every paired
    run;
  - no learning-rate or architecture search beyond the single bounded paired
    task-level calibration above. Loader-worker changes are allowed only before
    campaign sealing and must preserve membership, order, and optimizer
    semantics. A chunk-size change requires repeating equivalence and
    largest-bag preflights and relocking the campaign.
- A paired run fails if one branch skips a sample, produces nonfinite values,
  loses alignment, or cannot complete the locked protocol. Do not silently
  continue with only one representation.

## Metrics And Statistical Unit

### WSI diagnosis

- Primary: macro-F1 over WSIs.
- Secondary: balanced accuracy, ordinary accuracy, per-class precision/recall/
  F1, confusion matrix, and cross-entropy.
- Report `n_WSI` overall and per diagnosis for every split. Patch count is bag
  context, never the metric sample size.
- Report the single paired WSI run. Do not report optimization-seed standard
  deviation. LGSC and MC each have only two test WSIs, so class-specific
  findings remain exploratory.
- Compute a paired, diagnosis-stratified WSI bootstrap interval for the
  normal-minus-SO(2) metric difference. Each replicate resamples WSIs with
  replacement within each diagnosis while retaining the original class counts,
  so macro-F1 is always defined. This interval describes WSI-sampling
  uncertainty for the paired prediction difference, not optimization-seed
  uncertainty.

### Tissue classification

- Primary: macro-F1 over patches for every training-set size.
- Secondary: balanced accuracy, ordinary accuracy, per-class precision/recall/
  F1, confusion matrix, and cross-entropy.
- Report both `n_patch` and contributing `n_WSI` overall and per tissue class.
- Because patches from one WSI are correlated, compute normal-minus-SO(2)
  uncertainty with a paired WSI-cluster bootstrap, resampling WSIs and carrying
  all of their test patches together. Record the bootstrap seed and interval.
- Plot the label-efficiency curve against labelled training patches per class,
  showing the one paired normal/SO(2) point at each size. Do not draw
  training-seed error bars from different subset sizes.

No independent-patch significance claim is allowed for WSI diagnosis, and no
patch-level confidence interval may pretend correlated patches are independent.

## Outputs And Acceptance Artifacts

- Task-specific stable-selection CSVs and one audit binding their hashes,
  counts, class/WSI distributions, nesting, physical resolution, and split
  disjointness.
- One small readable paired trainer for the exact dual-T4 configuration and
  focused synthetic/local tests. Avoid a generic training framework.
- Per-run config, initialization hash, logical-manifest hashes, access
  transcript, train/validation history, selected-checkpoint identity, runtime,
  and final predictions.
- WSI attention metrics, confusion matrices, and coordinate-linked attention
  rankings. Mean pooling is an optional later control, not an initial artifact.
- Tissue metrics and confusion matrices for every nested subset plus the
  label-efficiency curve.
- A paired summary table comparing normal and SO(2) representations, including
  parameter count, runtime, the fixed initialization seed, and sample counts.
- Test outputs are written last after a gate verifies the development artifact
  set and its hashes. Development kernels mount only train/validation logical
  CSVs. A separate test-release kernel/config attaches the test logical CSVs
  and the sealed selected checkpoints only after that gate passes. No paper or
  issue update is automatic.

## Acceptance Criteria

1. All task CSVs and the physical-parts catalog are generated before training,
   byte-stable, hash-bound, and contain only valid physical locations shared by
   both representations. Part 11 resolves through the same catalog lookup as
   every earlier part.
2. No WSI crosses splits in either task. No coordinate crosses splits or is
   duplicated within one logical dataset.
3. WSI instance/bag files cover exactly the Spec 0022 target; every bag is
   complete and consumed as one WSI-labelled example.
4. Tissue full training has exactly 5,671 unique rows per class. Every smaller
   CSV has the exact per-class count in its filename and is contained in every
   larger CSV. Every eligible class-positive training WSI appears in every
   nested subset; per-WSI counts are audited. Validation/test bytes are
   invariant across all subset runs.
5. Normal and SO(2) paired runs use identical logical indices, initial classifier
   state, shuffled order, epoch ceiling, validation cadence, and training
   policy. All five tissue sizes share the same 30-epoch ceiling. Only latent
   values and their resulting gradients may differ within a paired run.
   Every midpoint and epoch-end validation/checkpoint is one atomic paired
   bundle whose two branches have the same completed epoch fraction and
   logical-order cursor. Early stopping cannot remove or freeze one branch; the
   pair stops only when both patience conditions are satisfied at the same
   boundary or the common 30-epoch ceiling is reached.
6. Train/validation access transcripts contain no test location. The test gate
   fails unless every development artifact and selected checkpoint hash is
   complete and no protocol field changed after training began.
7. Attention softmax covers all instances in each fixed bag. Full-bag and
   activation-checkpointed execution agree within locked forward and gradient
   tolerances for patch encoder, attention, and head on a synthetic bag; the
   maximum real bag passes dual-T4 forward/backward memory and timing preflight.
8. Metrics use the correct statistical unit and include required sample counts.
   Tissue uncertainty is WSI-clustered; WSI diagnosis never reports patch rows
   as independent samples.
9. Results retain all six predeclared paired configurations. No configuration
   may be rerun or discarded because its result is unfavorable. Nonfinite or
   misaligned paired runs fail visibly.
10. Focused tests, `git diff --check`, repo/workspace preflights, the Python
    quality gate after implementation, and independent clean-context review
    pass before any test result is promoted.
11. The sealed mount/capacity artifact proves the exact input topology, byte
    totals, largest-WSI forward/backward peak memory and step time, selected MIL
    execution mode and optional chunk size, deadline reserve, output budget, and
    separate test-release sequence before any real development launch.
12. The horizon package rejects any boundary other than confirmation-v1 epoch
    `2.0` at `2e-4`, restores the complete pair before training, commits exactly
    `318` further paired full-bag updates without a second warmup, constructs no
    tissue or test dataset, reconstructs and verifies the pre-resume access
    prefix, and writes all six epoch-`2.5` through epoch-`5.0` atomic boundaries.
    A finite but still insufficient-learning result exits successfully after
    publishing its audit and checkpoints; only a runtime, numerical, alignment,
    identity, or durability failure returns nonzero.

## Tests And Verification Commands

Completed local commands:

```bash
.venv/bin/python -m pytest -q tests/test_spec0023_supervised_manifests.py
.venv/bin/python -m pytest -q tests/test_spec0023_supervised_models.py
.venv/bin/python -m pytest -q tests/test_spec0023_mil_capacity_probe.py
.venv/bin/python -m pytest -q tests/test_spec0023_supervised_calibration.py
./scripts/kaggle_kernel.sh preflight-supervised-calibration sweep
./scripts/python_quality.sh
git diff --check
./scripts/agent_preflight.sh
```

The combined focused calibration/kernel slice passes 21 tests. The serial
sweep-package preflight passes its seven tests and byte-for-byte rebuild check.
The final repository Python quality gate passes with 958 tests passed, one
skipped, and clean formatting, Ruff, and basedpyright results. No Kaggle API or
remote kernel action is part of these checks.

Tests must cover WSI split leakage, duplicate/missing locations, model-specific
CSV substitution, unified part resolution including part 11, nested tissue
membership, class counts, bag gaps/overlap,
per-subset WSI representation, complete-bag attention, checkpointed-vs-full
forward and gradient equivalence, paired initialization/order, equal tissue
epoch ceilings and half-epoch validation, deterministic subset cycling, dynamic
resampling, exact midpoint/end boundary derivation for odd and even epoch
lengths, joint paired early stopping, atomic paired checkpoint/resume
equivalence, test-loader construction
before the gate, test access in development
transcripts, metric statistical units, cluster bootstrap grouping, and test-
artifact-last failure. The remote preflight additionally proves largest-bag peak
memory, step time, and the exact mount/session/output-capacity contract.
Focused calibration tests additionally pin exact 106/132 sweep endpoints, the
first nonzero warmup LR and `W=11/14`, every tissue subset's own epoch geometry,
the exact cosine endpoint, no cursor or parameter mutation on either-side
failure, fresh phase reset, validation-only confirmation inputs, and atomic
checkpoint/resume equivalence across warmup and half-epoch boundaries.
The continuation-focused tests additionally reject confirmation-v2 or any
wrong peak/cursor/hash, pin 318 constant-`2e-4` post-resume updates in epoch
orders 2-4, prove no second warmup, tissue construction, test input, chunked
bag, or deliberate negative-result failure path exists, verify the reconstructed
access prefix, and require all six resumable output fractions.

## Implementation Blockers

Replacement kernel version 2 completed and its train-only audit is preserved at
`runs/kaggle/ubc_ocean_supervised_calibration_sweep_v2`. Audit SHA-256
`a96be8110086ea5d7cf7526a10c120445c7d592ebf46f5d3dc36208382f7d9a2`
passes the strict validator against the exact sealed v2 config, manifests,
source tree, and private-input receipt. The legacy replay reproduces the v1
attempt-12 overflow and the corrected post-unscale replay commits all 12
attempts from the same initialization and order. Fresh tissue completes all
132 paired updates through `3e-3`. Fresh MIL commits 104 complete unchunked bags,
then attempt 105 at LR `0.0028413819922822273` produces one nonfinite
`head.weight` gradient element in each branch on the same 1,685-instance HGSC
bag; inputs, forward values, and every earlier named gradient are finite. The
last committed MIL LR is `0.0026911505420219024`.

This is not the original class-weight overflow: the new failure occurs after
post-unscale weighting with class weight `0.53`, at the high-LR end, and affects
both branches symmetrically. Both MIL EWMAs descend together by about 24% from
update 40/LR `8.32e-5` through update 60/LR `2.47e-4`; the human-sealed shared
MIL peak is therefore `2e-4`, more than fourteen times below the later failure.
Both tissue curves descend clearly; the shared tissue peak is `1.5e-3`,
conservative relative to the worse normal branch's minimum near `2.41e-3`.
The initial selection audit SHA-256 was
`387f932430ebd5546198ba88e8c7b06ddce35b4ff0a12f048bfaad0d52c99a16`;
the authorized MIL-only `2.5e-4` retry audit SHA-256 is
`d947d13f0835cadfe5fa8f25dd1e61daada8446c65aafa874df6d9076b98f2c8`.
Do not lower the scaler, broaden the search, chunk/truncate a bag, or inspect
test data. The historical sweep validates against its immutable launched spec
preimage (SHA-256 prefix `627f17...`); confirmation separately binds the current
spec and selection bytes.

Private confirmation kernel
`maximusshtefan/eqvae-ubc-ocean-supcal-confirmation` version 1 completed every
planned training/validation boundary but intentionally exited nonzero because
the combined audit status is `failed`. This is not a runtime or numerical
failure. Both MIL branches commit all 212 full-bag updates at shared peak
`2e-4`, have zero scaler skips, and publish complete paired checkpoints at
epochs 0.5, 1.0, 1.5, and 2.0. Their final EWMAs (`1.52913` normal,
`1.52817` SO(2)) exceed their end-of-warmup EWMAs (`1.31563`, `1.31870`), and
their whole-epoch mean weighted losses rise by about 3% in epoch two. MIL
validation loss finishes near `1.527` for both branches; macro-F1 briefly rises
to about `0.245` at epoch 1.5 but ends near `0.177`/`0.173`. Therefore both
branches fail the locked learning check. Insufficient learning is not eligible
for the numerical half-peak fallback and does not authorize a broader search.

Tissue passes at shared peak `1.5e-3`: both branches are finite with zero scaler
skips, final EWMAs are `0.34836`/`0.32171`, and final validation macro-F1 is
`0.69728`/`0.75767`. The downloaded audit SHA-256 is
`ec7a3554e4864b328dd86485c30eef6d5a726fc014acf89012bca07aea529571`;
all eight boundary manifests match their checkpoint/progress hashes and the
final MIL checkpoint retains both models, optimizers, GradScalers, RNG, order,
validation, patience, access, and campaign state. No test artifact was mounted
or accessed. The resulting explicit scientific decision authorized the single
MIL-only retry at `2.5e-4`, with tissue frozen and the same 10% warmup. That
retry is now complete; do not add another LR trial or inspect test data.

Private input dataset version 2 is byte-verified. Private kernel
`maximusshtefan/eqvae-ubc-ocean-supcal-confirmation` version 2 completed all
212 paired full-bag updates and four half-epoch boundaries at `2.5e-4`, then
deliberately exited nonzero because MIL again reports `insufficient_learning`.
Both branches are finite with zero scaler skips. Final EWMAs are
`1.52739`/`1.52648` versus warmup-end `1.34211`/`1.34510`; epoch-two mean loss
rises `2.47%`/`2.21%`. Final validation loss is `1.53395`/`1.53281`, final
macro-F1 is `0.08276`/`0.13122`, and the best macro-F1 is `0.18333` for both.
The audit SHA-256 is
`0e637fafbd62e861d64a1fe9b9977fb15d51bccaa3a13521c85817f60b6c5668`.
All four boundary manifests hash-verify and the epoch-2 checkpoint retains both
models, optimizers, scalers, RNG, order, validation, patience, access, and
campaign state. Tissue was not executed and no test data was mounted or
accessed. Do not try another LR. The explicit horizon decision now authorizes
only the hash-bound version-1 `2e-4` continuation through epoch 5 described
above; it does not authorize the later architecture stages yet.

The local synthetic preflight is intentionally bounded: the focused model,
reader, pairing, scheduling, package, and checkpoint tests finish in a few
seconds on the local CPU. It proves ordinary part-11 resolution, grouped
read restoration, global complete-bag attention, checkpointed/full forward and
all-parameter gradient equivalence, independent identical paired
initialization, and nested epoch ordering. It does not claim to measure the
8,149-instance real bag or dual-T4 MIL capacity. That separate real probe is now
complete at Kaggle kernel version 2. Its compact artifact SHA-256 is
`8096dab0f0ff350a0d6508e54b0abae54d11d43ba292bf22569ded3156044acb`.
With both scalers initialized to 32,768, both the first and second steps used all
8,149 attention entries, produced finite gradients without a scaler skip, and
the settled step completed in 0.177 seconds paired wall time. Peak
allocated/reserved memory was 2.092/2.963 GB for both branches, leaving about
12.5 GB free per 15.64 GB T4. The full probe, including Torch bootstrap and
mounts, used 160.8 seconds and preserved 28,639.2 seconds of the 8-hour session.

## Adversarial Review Result

The first independent Terra pass found three blockers: unstratified tissue
prefixes, unspecified MIL chunk gradients, and no Kaggle capacity/session
contract. It also required paired WSI uncertainty, a label-free top-up input,
and clearer status handling. The spec now uses WSI-stratified nested priorities,
exact activation-checkpoint semantics and equivalence preflight, sealed mount/
session planning, paired diagnosis-stratified bootstrap, and separate test
release. The clean-context re-review found no remaining scientific or
implementation blocker. A later focused review of the reduced six-run campaign
rejected different seeds per tissue size. After considering equal-compute and
equal-exposure alternatives, the user selected a simpler fixed-epoch protocol:
both tasks and all five tissue sizes have a 30-epoch ceiling. Every tissue epoch
drops its partial tail as required by the locked loader contract.
The independent capacity-probe review then caught a missing dedicated Kaggle
push guard, incomplete deadline/output evidence, and an OOM path that began too
late to catch transfer failure. Those issues were fixed before submission; the
reviewer rechecked the rebuilt guard and runner and found no remaining launch or
scientific blocker.

The independent clean-context calibration review then challenged the local
implementation for representation-dependent treatment, split leakage,
mismatched paired order or LR, incorrect epoch semantics, accidental bag
truncation, unnecessary abstractions, and incomplete resumability. It found
five blockers: insufficient binding to the canonical manifest audit, a path
that could accept one WSI split across multiple bags, a free-form sweep binding,
an incomplete numerical-failure exit/fallback path, and resume logic that had
not proved continuation from the committed boundary. The implementation now
binds the canonical manifest hashes, rejects duplicate WSI bags, requires the
downloaded sweep audit and exact sweep-config preimage before confirmation,
fails a numerically unsuccessful confirmation after its sole half-peak
fallback, and verifies/reloads every paired boundary before continuing. A final
Terra re-review found no remaining P0/P1 issue in the remediation.

## Known Risks

- There are only 106 training and 23 test WSIs; attention MIL can overfit and
  rare diagnosis test results have high uncertainty. Keep the architecture
  small and conclusions bounded.
- The largest real 8,149-map bag fits unchunked with about 12.5 GB free per T4.
  Keep the proved unchunked execution; checkpointing remains only a fallback if
  a future locked architecture or precision policy changes the capacity premise.
- High-purity tissue masks omit mixed and unpainted tissue. Results measure the
  defined three-class high-purity task, not full-slide segmentation ability.
- Balanced tissue training changes class priors. Validation/test metrics must
  use their natural fixed distributions and include confusion matrices.
- Shared hyperparameters may not be individually optimal for either
  representation, but model-specific tuning would answer a different question.
- Each initial configuration has one optimization trajectory. Shared tissue
  initialization/order isolates the nested label-count comparison, but
  seed-to-seed variance still requires a later repeat of the full predeclared
  campaign.
- Attention weights can identify influential coordinates but are not calibrated
  probabilities or causal explanations.

## Adversarial Checks

- Give the two models different rows, order, initialization, early-stop rule,
  or optimizer settings while preserving nominal counts.
- Let physical shard mixing leak one test WSI into a training bag.
- Dynamically resample patches from the full candidate pool each epoch.
- Build a smaller tissue CSV independently instead of as a prefix.
- Let a small tissue prefix omit a class-positive training WSI.
- Report patch count as WSI diagnosis `n` or use an independent-patch bootstrap
  for correlated tissue rows.
- Normalize attention over chunks rather than the complete WSI.
- Detach chunk tokens and silently train only the attention/head.
- Tune one model on test or rerun only an unfavorable configuration.
- Apply an ordinary image rotation to SO(2) latent channels as augmentation.

## Open Questions

The synthetic transformer capacity check above passed. Its result does not
authorize a learning run. Integral attention, convolution
changes, and any wider search remain deferred; tissue remains frozen. A real
transformer learning diagnostic requires its own explicit protocol and user
authorization before launch. Test data remains sealed throughout development.
The user retains two layers, one CLS, and eight registers. Spec 0022 governs
the largest training WSI45630's missing-patch extraction solely for a later
full-bag memory-fit check; authorized execution status lives in `CURRENT.md`.
The current 25%-coverage learning bags remain unchanged. No depth change or
full-coverage learning run is implied.

Remote horizon execution: private dataset
`maximusshtefan/eqvae-ubc-ocean-mil-horizon-inputs` version 1 is verified by
receipt SHA-256
`cb011a1c60109629ccbd7f6ff49a4decd9408a7d1a1c351b9df07cac756fb86b`.
Private kernel `maximusshtefan/eqvae-ubc-ocean-mil-horizon` version 1 completed
on 2026-08-28. Its submitted config/wrapper SHA-256 values are
`3f0ad78a8c33340e062f12daf106dc39a824c41e590f080cb89f4d8a6e85ea57`
and
`6e079f649655e12fa39a549af3a7c4c20de400d24392c08f1da6c8a382ad582c`.
The downloaded audit SHA-256 is
`2edeb5d20471ebc559e3bb7f6f7cde4aeaa06723b98bdcfc28a7dca4039525fb`.
It reports 318 additional paired full-bag updates, zero scaler skips, no
repeated warmup, no chunking, and six valid boundaries through epoch 5. Both
epoch-5 mean losses are below epoch 2 (`1.63017 < 1.73726` normal and
`1.62018 < 1.73156` SO(2)), but final EWMAs rise to `1.86584` and `1.90369`.
Neither branch improves its pre-resume best validation macro-F1: post-resume
bests are `0.24396` and `0.24444`, versus `0.24500` and `0.24644`. The locked
result is therefore `insufficient_learning`; this is scientific evidence, not
a numerical or transport failure.

Remote width-128 execution: private dataset
`maximusshtefan/eqvae-ubc-ocean-mil-width128-inputs` version 1 is byte-verified
by receipt SHA-256
`cbf128e082de5b6d662df5423e2c1b25a0b6316cb578ff30ba98b8ccc5217848`.
Private kernel `maximusshtefan/eqvae-ubc-ocean-mil-width128` version 1 was
accepted on 2026-08-28 and completed. Its submitted config/wrapper
SHA-256 values are
`b2de025ca3b647a2c9feecb640115bd60ea9624c76c4587de631ae166b691dde`
and
`ca7b902803ce973cb96adeec7462f2d64f97c5619d66d4907fc13b8854f9b98f`.
That immutable version predates the subsequent semantic optimizer-group
correction and therefore retains the historical single AdamW group with
`1e-4` decay on all parameters. Do not relabel it. At its peak LR and 530
updates, the largest possible cumulative difference on a newly zero-decay 1D
parameter is only about `1.05e-5` relative, so version 1 is not restarted.
All later supervised packages use the locked matrix-only decay rule above; if
width 128 is selected, its corrected-group stability is checked before the
full campaign rather than creating an optimizer search.

The downloaded compact audit is
`runs/kaggle/ubc_ocean_mil_width128_v1/spec0023_supervised_calibration_audit.json`,
SHA-256 `3e0933d87e334ba977dfaaca45bab99b70984cd88a1fa0f0097f9259bb77de23`.
It validates the exact submitted config and records 530/530 paired complete-bag
updates, ten validation checks per branch, zero scaler skips, no chunking, and
train/validation-only access. Its scientific result is
`insufficient_learning`. Epoch-five mean weighted losses are
`1.6298536`/`1.6217492` for normal/SO(2), effectively unchanged from width 64's
`1.6301738`/`1.6201767`. Best validation macro-F1 is
`0.2439628`/`0.2464396`, versus width 64's `0.2450000`/`0.2464396`; neither
branch improves. Kaggle's remote inventory contains all ten half-epoch
`manifest.json`/`paired_checkpoint.pt`/`progress.json` triplets, each of which
the runner load-verified before continuing. The current Kaggle CLI bulk output
download omits those repeated nested filenames locally, so the audit and remote
inventory—not a nonexistent local checkpoint directory—are the retained proof.
Increasing only the gated-attention scorer width therefore fails this stage;
the next authorized architecture diagnostic is class-specific gated attention,
not another width or LR trial.

Class-specific gated-attention diagnostic contract:

- Add only the one-off `class_specific` package mode, private input dataset
  `maximusshtefan/eqvae-ubc-ocean-mil-class-attn-inputs`, and private kernel
  `maximusshtefan/eqvae-ubc-ocean-mil-class-attn`. The input bundle contains
  the current source tree and the same WSI train/validation catalog/manifests as
  width 128; tissue, test, resume, and latent payload copies remain absent.
- Keep the width-128 patch encoder and gated features. For complete-bag tokens
  `H` of shape `[N,128]`, compute five score columns
  `S = attention_w(tanh(attention_v(H)) * sigmoid(attention_u(H)))` of shape
  `[N,5]`, then normalize each diagnosis independently across all `N` patches:
  `A = softmax(S, dim=0)`. Thus every column of `A` sums to one over the full,
  unchunked bag. Compute the five diagnosis-specific pooled tokens with one
  matrix multiplication, `Z = A^T H`, shape `[5,128]`.
- Retain the existing `Linear(128,5)` head. Evaluate `head(Z)` once, producing a
  `[5,5]` matrix, and take its diagonal as the five logits. Equivalently,
  logit `c` is the dot product between pooled token `Z[c]` and head row `c` plus
  bias `c`; cross-class off-diagonal scores are not part of the objective. This
  is diagnosis-specific gated attention, not Transformer self-attention or a
  generic configurable multihead module.
- Start fresh from seed `1701`. Copy the exact width-128 initialization for the
  patch encoder, `attention_v`, `attention_u`, and head. Repeat the width-128
  scalar `attention_w` weight and bias into all five score rows. At update zero,
  all five attention columns and pooled tokens are identical and the diagonal
  head result exactly equals the width-128 logits; the only new capacity is the
  ability of the five scorer rows to specialize during learning. Both normal
  and SO(2) branches receive byte-identical independent copies of this state.
  The downloaded width-128 paired initialization SHA-256 is
  `cfef73d525bf5e58501ca7aa9c93ed65a9a069f08ebf2fcefea63687b04dbcbf`;
  the deterministic five-row lifted paired initialization SHA-256 is
  `d6f916aafd6b544d62c2f481ff5f619e6485156dac08461b96546be8b5717c22`
  and the runner verifies it before opening training stores.
- Keep the corrected AdamW grouping (`ndim >= 2` decay `1e-4`, `ndim < 2`
  decay zero), shared peak `2e-4`, 11-update zero-to-peak warmup, five fresh
  epochs, class weights, complete-bag order, FP16/GradScaler settings, and ten
  half-epoch validation/resumable-checkpoint boundaries unchanged. A finite
  `insufficient_learning` result is still published successfully. Report all
  epoch mean/EWMA losses and validation points, and whether each branch exceeds
  the width-128 best macro-F1 (`0.2439628` normal, `0.2464396` SO(2)). Do not
  inspect test data, change LR, run another width, or auto-start the later
  integral/learned-gate stages.
- The exact reference is downloaded width-128 audit SHA-256
  `3e0933d87e334ba977dfaaca45bab99b70984cd88a1fa0f0097f9259bb77de23`.
  The local package builder must read and fail-closed validate that exact audit,
  including its width-128 identity, exact submitted config, 530-update/ten-
  validation-check geometry, WSI-only logical manifests, complete-bag/no-chunk
  policy, and zero scaler skips, then bind its hash and reference best metrics
  into the class-specific config. The audit is predecessor evidence, not a
  learning input, and must not be placed in the remote input dataset.
  Every class-specific boundary additionally stores both branches' training
  histories and current EWMA values alongside the already-required model,
  optimizer, GradScaler, RNG, order, validation, best/patience, access, schedule,
  and campaign state. `load_paired_boundary` returns that payload for a later
  separately authorized continuation; do not build a campaign manager or launch
  a resume kernel preemptively.
  Local implementation and focused tests do not authorize publishing the input
  dataset or launching the kernel; each remote write still requires explicit
  user authorization.

Local implementation status: `ClassSpecificAttentionMILClassifier` and its
function-preserving width-128 lift are complete; the one-off
`class_specific` input/package/runner route and checkpoint training-history/EWMA
state are implemented. Focused Ruff, BasedPyright, shell/Python syntax, and eight
model/package/checkpoint tests pass. Independent clean-context review found no
remaining P0/P1 after the exact predecessor/lifted initialization hashes and
the perturbed-row diagonal-algebra test were added. The final local input bundle
is built from this spec/source state and contains only the WSI train/validation
logical assets plus source; it has no tissue, test, resume, or latent binary
asset. Private version 1 ran 465 successful paired updates and then failed on
attempt 466 (WSI `55287`, 3,869 patches). Both inputs and forward paths were
finite and the SO(2) gradients were finite, but the normal branch overflowed in
the patch encoder/attention backward path at loss scale `32768`; its head
gradients remained finite. The downloaded failure audit SHA-256 is
`07fd04096c151fbd0be2b2b997e7f3c62dad9cdc84a0b421d6eec4a175ad9efc`.
All eight boundaries through epoch 4.0 are hash-verified. This is not OOM, bad
latent data, bag truncation, or a nonfinite forward loss.

Class-specific paired loss-scale correction:

- Add only the one-off `class_specific_scale_fix` package using private WSI-only
  input dataset `maximusshtefan/eqvae-ubc-ocean-mil-class-scale-inputs` and
  private kernel `maximusshtefan/eqvae-ubc-ocean-mil-class-scale`. The builder
  must authenticate the exact version-1 failure audit above, including 465
  committed updates, failed attempt 466, scale `32768`, finite inputs/forwards,
  normal-only nonfinite encoder/attention gradients, complete bags, and
  train/validation-only access. The audit is provenance, not a learning input.
- Start a fresh five-epoch run with the identical seed-1701 classifier state,
  width 128, five attention maps, data/order, class weights, AdamW groups, peak
  `2e-4`, 11-update warmup, and half-epoch boundaries. Initialization is not
  changed: convolution weights remain Kaiming-normal, linear weights Xavier,
  GroupNorm scales one, and every bias zero. This preserves the controlled
  architecture comparison and the exact update-zero width-128 function.
- Both GradScalers start at `32768` with growth interval `1_000_000`. Across the
  entire run, permit at most one synchronized paired backoff. If either branch
  has a nonfinite raw gradient immediately after unscale, before the FP32 class
  weight is applied, apply no optimizer update to either branch, halve both
  scales to the same value, discard both gradient sets, and retry the same
  already-loaded complete WSI at the same LR without advancing the successful-
  update cursor or data order. Check finiteness again after applying the class
  weight, but treat a weight-induced FP32 overflow as terminal because changing
  the AMP scale cannot repair it. If the raw-gradient retry or any later update
  overflows, fail. Do not lower only one branch's scale, skip only one optimizer,
  change LR, truncate/chunk the bag, or consume a different example.
- Record the backoff's successful-update position, both pre/post scales, and
  global used/remaining budget in the audit and every later resumable boundary.
  A discarded backward attempt is not a successful update. The final result
  still requires 530 committed paired updates and ten validations. Tissue and
  sealed test data remain absent, and this correction does not authorize the
  integral-attention or equivariant-convolution ablations.
- Local implementation/build/preflight does not authorize a Kaggle dataset or
  kernel write. Remote publication and launch still require separate explicit
  user authorization.

## Related Files

- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0019-task-consumption-manifests-and-kaggle-work-plan.md`
- `docs/specs/0020-fp32-latent-shard-format-and-io.md`
- `docs/specs/0021-dual-model-wsi-latent-inference.md`
- `docs/specs/0022-additive-cancer-coverage-topup.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
