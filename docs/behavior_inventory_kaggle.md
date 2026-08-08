# Kaggle Behavior Inventory

Status: current historical inventory
Last updated: 2026-06-11

This file records the behavior of the historical Kaggle notebooks before the
repo extracts reusable modules for the translatable normal VAE baseline.

Sources inspected:

- `kaggle/train_runs`
- `kaggle/dataset_generation`
- `kaggle/generate_dataset_Classification_With_Masks`

This is an inventory, not an implementation contract. Spec 0001 decides what is
carried forward into the new comparable non-equivariant VAE.

## Confirmed Kaggle Sources

The following slugs were checked through the Kaggle CLI on 2026-06-06. Do not
replace them with display names from the web UI.

| Source | Confirmed slug | Historical role |
| --- | --- | --- |
| UBC-OCEAN competition | `UBC-OCEAN` | Raw labels, WSI images, and thumbnails for dataset generation. |
| Supplemental masks | `sohier/ubc-ovarian-cancer-competition-supplemental-masks` | Identifies the UBC-OCEAN WSIs with available non-exhaustive masks. These masked WSIs are intentionally held out from train/validation for the autoencoder test set and later supervised experiments. |
| Raw atlas | `maximusshtefan/raw-atlas-ubc-ocean` | Historical atlas input before train/valid split. |
| Train/valid atlas | `maximusshtefan/train-val-atlas-ubc-ocean` | Historical split atlas used to generate patch shards. |
| Pre-shuffled patches | `maximusshtefan/patches-pre-shuffled-ubc-ocean` | Historical pre-shuffled train and validation patch binaries. Use this as the first CLI kernel dataset source. It does not contain a held-out test shard. |
| Non-eq-VAE output | `maximusshtefan/non-eq-vae-output` | Historical FSQ checkpoint/output source for resume only. Do not attach it to the new normal VAE baseline unless intentionally reproducing the old run. |

The current CLI-managed debug kernel should start with:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
```

Dataset-generation kernels, if reintroduced later, should use
`competition_sources` for `UBC-OCEAN` and explicit `dataset_sources` for the
mask/atlas inputs.

The pre-shuffled dataset file list was checked through the Kaggle CLI on
2026-06-10. It contains:

```text
dataset/ubc_train_shuffled.bin
dataset/ubc_train_shuffled.csv
dataset/ubc_ocean_valid.bin
dataset/ubc_ocean_valid.csv
```

It does not contain `test` files. Final paper claims need a sealed held-out test
set generated and documented separately from the train/validation tuning loop.

The accessible UBC-OCEAN source listings were rechecked through the Kaggle CLI
on 2026-06-10 with `--page-size 200`. The official competition listing contains
538 train WSI images, 513 train thumbnails, 25 TMA train images without
thumbnails, one public `test_images/41.png` stub, `train.csv`, `test.csv`,
`sample_submission.csv`, and `updated_image_ids.json`. The supplemental mask
dataset contains 152 mask PNGs. Every mask filename maps to an official
non-TMA train image ID.

Small CSV/JSON metadata files inspected on 2026-06-10:

- official `UBC-OCEAN/train.csv`;
- official `UBC-OCEAN/test.csv`;
- official `UBC-OCEAN/updated_image_ids.json`;
- `patches-pre-shuffled-ubc-ocean/dataset/ubc_train_shuffled.csv`;
- `patches-pre-shuffled-ubc-ocean/dataset/ubc_ocean_valid.csv`.

The derived patch split was verified against those metadata files:

| Split or pool | WSI count | Patch rows | Notes |
| --- | ---: | ---: | --- |
| Pre-shuffled train | 322 | 300000 | Non-TMA, no supplemental-mask WSI overlap. |
| Pre-shuffled validation | 39 | 30000 | Non-TMA, no supplemental-mask WSI overlap. |
| Masked holdout candidate pool | 152 | not generated yet | All remaining non-TMA supplemental-mask WSIs. |
| Official TMA images | 25 | not used | All excluded from the current patch split. |

The 322 train WSIs, 39 validation WSIs, and 152 masked candidate WSIs partition
the 513 official non-TMA train WSIs. Train/validation WSI overlap is zero.
Train/validation overlap with the 152 supplemental-mask IDs is zero. The exact
masked holdout candidate list is stored in:

```text
docs/data/ubc_ocean_masked_holdout_ids.csv
```

Masked holdout candidate label counts from official `train.csv`:

| Label | WSI count |
| --- | ---: |
| CC | 33 |
| EC | 36 |
| HGSC | 56 |
| LGSC | 15 |
| MC | 12 |

The patch CSV numeric labels map to official labels as:

| Patch label | Official label |
| ---: | --- |
| 0 | CC |
| 1 | EC |
| 2 | HGSC |
| 3 | LGSC |
| 4 | MC |

The private derived datasets `maximusshtefan/raw-atlas-ubc-ocean` and
`maximusshtefan/train-val-atlas-ubc-ocean` expose metadata through the CLI and
are marked private, but `kaggle datasets files` returned 403 for their file
lists when last checked on 2026-06-10. Treat that as last-observed operational
status rather than durable provenance. Do not rely on those file lists until
access is fixed or the needed files are regenerated from documented inputs.

User-confirmed split intent, recorded 2026-06-10:

- UBC-OCEAN includes a subset of WSIs with supplemental masks.
- Those masks are not exhaustive over the full WSI, so unmasked regions inside a
  masked WSI must not be treated as exhaustive negative/normal labels.
- The current train/validation patch dataset was made from WSIs without
  supplemental masks.
- WSIs with supplemental masks were deliberately left out of train/validation so
  they can become the held-out autoencoder test set and support later supervised
  experiments.
- The test-set generator must therefore select the masked-WSI pool, preserve
  slide-level separation, and document the exact image ID list.

The binary file sizes imply exact patch counts after subtracting the 64-byte
header and dividing by `3 * 256 * 256` bytes:

| File | Patches |
| --- | ---: |
| `dataset/ubc_train_shuffled.bin` | 300000 |
| `dataset/ubc_ocean_valid.bin` | 30000 |

## Historical Training Notebook

Notebook source:

```text
kaggle/train_runs
```

Kaggle metadata:

- language: Python notebook
- Python version recorded by the notebook: 3.12.12
- accelerator: NVIDIA Tesla T4
- GPU enabled
- internet enabled in the historical notebook
- execution command cell:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
torchrun --standalone --nproc_per_node=2 train.py
```

The notebook serializes a large `train.py` file through `%%writefile` and then
runs that file with two DDP ranks. The new repo workflow must invert that: repo
code and generated launchers are the source of truth, and Kaggle only receives a
script kernel.

### Historical Training Inputs

The notebook referenced train and validation paths that are present in the
current pre-shuffled dataset:

```text
/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_train_shuffled.bin
/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_train_shuffled.csv
/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_ocean_valid.bin
/kaggle/input/datasets/maximusshtefan/patches-pre-shuffled-ubc-ocean/dataset/ubc_ocean_valid.csv
/kaggle/input/datasets/maximusshtefan/non-eq-vae-output/run_004/checkpoints/checkpoint_ep_28.5.pt
```

These files are the train/validation input contract for the first baseline. They
are not a full train/validation/test contract, because there is no held-out test
shard in this dataset.

For the new baseline, keep dataset roots configurable. Do not hard-code the
historical `/kaggle/input/datasets/...` shape unless the launcher verifies that
Kaggle still mounts the dataset that way.

### Historical Config

Important values from `CONFIG`:

- `run_name`: `run_005`
- `debug_mode`: `True`
- `resume_checkpoint`: FSQ checkpoint from `non-eq-vae-output`
- `batch_size`: 60
- `num_workers`: 1
- `epochs`: 35
- `lr`: `5e-4`
- `weight_decay`: `1e-5`
- `corrupt_prob`: 0.30
- `corrupt_alpha`: `[0.75, 1.25]`
- `corrupt_beta`: `[-0.10, 0.10]`
- `corrupt_noise`: 0.03
- `fsq_levels`: 16
- `latent_dim`: 16
- `output_dir`: `/kaggle/working`

The new normal VAE should not inherit `fsq_levels`, discrete-index telemetry, or
the FSQ resume checkpoint.

### Historical Patch Loader

The historical loader:

- reads only `wsi_id` from the CSV to count samples;
- uses a 64-byte binary header;
- memory maps the binary file with sequential advice when available;
- interprets patch data as `uint8` with shape `(-1, 3, 256, 256)`;
- returns `torch.from_numpy(...)`;
- divides data by DDP rank using contiguous slices;
- requires `samples_per_rank % batch_size == 0`;
- does not use an explicit sampler or shuffle inside the training loop.

The new data module should preserve the patch binary contract initially, but it
must make split/leakage checks explicit and should not depend on silent
contiguous-rank assumptions unless they are tested.

### Historical Normalization And Corruption

Training converts `uint8` CHW tensors to:

```text
x = images.float() / 127.5 - 1.0
```

The denoising input is generated by a HED-space stain/illumination corruptor:

- converts sRGB `[-1, 1]` to linear RGB;
- converts to optical density and then HED-like channels;
- samples per-image/channel alpha and beta perturbations;
- uses the configured wide ranges for H and E;
- uses small fixed D-channel ranges: alpha `[0.98, 1.02]`, beta `[-0.01, 0.01]`;
- reconstructs back to sRGB;
- adds spatial Gaussian noise in image space;
- applies corruption with a per-image Bernoulli mask using `corrupt_prob`.

The new comparison should use the same corruption policy for both branches once
locked, and should log whether the target is clean `x_clean` and the input is
`corrupt(x_clean)`.

Clean-context implementation audit resolution: the relocked
`src/eqvae/corruption/stain.py` satisfies the historical audit and the historical
corruptor is not copied directly. Its HED matrices (`RGB_FROM_HED`/`HED_FROM_RGB`)
match scikit-image 0.26.0 and are verified true inverses, applied channel-first
via `einsum("bchw,cd->bdhw", ...)`; the conversion buffers are unambiguously named;
and the historical corruption used rank-invariant per-sample seeding — a per-sample
`torch.Generator` seeded by
`blake2b(corruption_seed, split, semantic_sample_key, step, view, version)`, with no
global RNG. This describes the retired deterministic implementation, not a current
requirement. The speed-first runtime uses native compile-friendly RNG; bitwise and
per-sample reproducibility are explicit non-objectives.

### Historical Model

The historical model is an FSQ autoencoder, not a normal VAE:

- encoder/decoder autoencoder;
- finite scalar quantizer with `latent_dim=16` and `fsq_levels=16`;
- no Gaussian posterior, no sampling reparameterization, and no KL term;
- 7x7 stem, 3x3 blocks, 1x1 projections/shortcuts;
- GroupNorm, ReLU/SiLU-style nonlinearities;
- PixelShuffle/sub-pixel upsampling in the decoder;
- discrete-code and bin-histogram artifacts.

These behaviors are useful historical evidence but are not allowed in the first
translatable baseline unless a future spec explicitly changes the contract.

### Historical Objective

The historical objective combines:

- Charbonnier reconstruction loss;
- SSIM term with default weight 0.1;
- FSQ excursion penalty.

It does not implement the normal VAE objective. There is no KL loss and no
posterior statistics. Spec 0001 deliberately replaces this with
`L1 + 0.1 * (1 - SSIM) + beta * KL`; SSIM remains a reported metric but is also
part of the first-run reconstruction objective.

### Historical Training Loop

The historical loop uses:

- NCCL DDP with two ranks;
- `torch.compile` around the step function;
- PyTorch compiled autograd enabled;
- AMP autocast with `float16`;
- CUDA GradScaler;
- AdamW with custom parameter groups;
- gradient clipping with `max_norm=1.0`;
- cosine warm restarts;
- manual warmup over the first 5 percent of total steps;
- checkpointed torch, CUDA, and NumPy RNG state.

Debug mode runs only one relative epoch and breaks after a few train/valid
batches. The new local tests should exercise CPU shape/compile/precision
contracts without requiring a GPU; Kaggle remains the GPU execution surface.

Historical precision/autograd behavior:

- The main compiled training step ran the model forward under CUDA autocast with
  `dtype=torch.float16` and `cache_enabled=False`.
- The encoder final projection explicitly disabled autocast and cast to FP32
  before `enc_norm -> enc_act -> enc_out`.
- The FSQ quantizer explicitly disabled autocast, cast latents to FP32, and used
  straight-through `.detach()` terms around clamp/round operations.
- The decoder final `tanh` was computed after casting logits to FP32.
- The HED/OD corruptor ran under `torch.no_grad()`, converted to FP32 for
  log/exp/color operations and random draws, then returned a clamped tensor in
  the input dtype.
- The reconstruction objective was computed in FP32 outside autocast after
  casting reconstructions to FP32.
- Metric accumulation, FSQ codebook telemetry, validation, fixed-patch artifact
  generation, and equivariance artifacts ran under `torch.no_grad()`.
- The branchless validation path still called the corruptor with probability
  `0.0`, so it consumed corruption RNG even though the clean input was selected.
  Spec 0001 intentionally improves this: clean validation must not call the
  corruptor or consume corruption RNG.

### Historical Resume

Resume behavior:

- loads a checkpoint from the previous FSQ output dataset;
- restores model, optimizer, scheduler, scaler, and RNG state;
- treats checkpoint names containing `.5` as midpoint checkpoints;
- fast-forwards by wrapping the dataset in `torch.utils.data.Subset`;
- writes `checkpoint_ep_*`, `best_model.pt`, and keeps a short checkpoint
  history.

The new baseline needs save/resume utilities, but it must not resume from an
FSQ checkpoint.

### Historical Metrics

Logged metrics include mean and standard deviation for:

- loss;
- Charbonnier;
- SSIM;
- MSE;
- MAE.

The historical CSV does not log per-image metric rows or an explicit sample
count `n`. The new evaluator must report SSIM, MAE, MSE, and PSNR with mean,
standard deviation, and `n`, and it must support boxplots.

### Historical Artifacts

Artifacts include:

- `hyperparameter_config.json`;
- fixed original reference patches;
- `train_dynamics.csv`;
- `epoch_stats.csv`;
- continuous latent `.npy` arrays;
- discrete index `.npy` arrays;
- reconstructions;
- FSQ bin histogram `.npz`;
- checkpoints and `best_model.pt`.

The historical qualitative protocol keeps a fixed validation batch and saves up
to 25 patches.

### Historical Rotation/Latent Check

The old notebook performs a small qualitative equivariance check:

- uses the fixed validation batch;
- uses only `min(batch_size, 25)` patches;
- applies `torch.rot90` for 90, 180, and 270 degrees;
- compares continuous latent maps from rotated inputs against rotated latent
  maps from clean inputs;
- saves reconstructions from the rotated input path and transformed-latent path;
- logs `equivariance_error_25_patches`.

This is not a continuous `SO(2)` evaluation and is not sufficient for the new
paper claim. The new route should still keep a fixed 25-patch qualitative set,
but it must add fixed continuous angles and the required
`rotated_input_vs_latent_grid.*` style artifacts.

## Historical Dataset-Generation Notebook

Notebook source:

```text
kaggle/dataset_generation
```

Kaggle metadata:

- language: Python notebook
- Python version recorded by the notebook: 3.12.12
- accelerator: none
- internet enabled in the historical notebook
- installs `libvips` and `pyvips` inside Kaggle

Historical inputs:

```text
/kaggle/input/UBC-OCEAN/train.csv
/kaggle/input/UBC-OCEAN/train_images
/kaggle/input/UBC-OCEAN/train_thumbnails
/kaggle/input/raw-atlas-ubc-ocean/raw_atlas.csv
/kaggle/input/train-val-atlas-ubc-ocean/train_val_atlas.csv
/kaggle/input/ubc-ovarian-cancer-competition-supplemental-masks
```

Important dataset-generation values:

- `PATCH_SIZE`: 256
- `VAL_SPLIT_RATIO`: 0.1
- `TRAIN_TARGET_PER_CLASS`: 60000
- `VAL_TARGET_PER_CLASS`: 6000
- `TOTAL_PARTS`: 6 for train, 1 for valid
- `NUM_WORKERS`: 3
- train/valid split uses `random_state=42` and label stratification;
- outputs binary patch shards and metadata CSVs.

Binary output contract:

- filenames: `ubc_ocean_{suffix}.bin`, `ubc_ocean_{suffix}.csv`;
- metadata columns from the generator: `idx,wsi_id,label,x,y`;
- current uploaded pre-shuffled CSVs differ slightly: `ubc_train_shuffled.csv`
  has `wsi_id,label,x,y`, while `ubc_ocean_valid.csv` has
  `idx,wsi_id,label,x,y`; future loaders must key by column name and treat
  `idx` as optional;
- header format: `<8sIQiiii3s25x`;
- magic: `UBC_DATA`;
- header size: 64 bytes;
- patch layout: CHW;
- dtype: `uint8`;
- patch shape: `3x256x256`;
- checksum: CRC32 over written data.

The new repo should not regenerate data casually. If data generation is needed,
write a separate spec and preserve the atlas/split/checksum contract.

## Classification/Validation/Test Dataset Notebook

Notebook source:

```text
kaggle/generate_dataset_Classification_With_Masks
```

This notebook was pushed from Kaggle on 2026-06-10 and is relevant because it
captures the train/validation shard generation route and the current attempt to
create a held-out test set for the autoencoder. The same held-out test set can
later serve the supervised classification experiment. It is still historical
notebook evidence, not canonical repo implementation.

Its metadata records:

- language: Python notebook;
- Python version recorded by the notebook: 3.12.12;
- accelerator: none;
- internet enabled in the historical notebook;
- data sources include the `UBC-OCEAN` competition and the supplemental-mask
  dataset source.

Important behavior:

- installs `libvips` and `pyvips`;
- sets `TMPDIR`, `TEMP`, `TMP`, and libvips cache variables to avoid Kaggle
  temporary-storage failures;
- contains a debug-path selection that builds `with_mask` from the supplemental
  masks dataset and excludes those image IDs plus TMA slides from the train/valid
  candidate pool, matching the intended train/valid versus masked-test split;
- can create a leakage-resistant train/valid atlas by splitting slides, not
  patches, with `random_state=42`;
- samples `TRAIN_TARGET_PER_CLASS = 60000` and
  `VAL_TARGET_PER_CLASS = int(0.1 * TRAIN_TARGET_PER_CLASS)`;
- writes `split = 'train'` and `split = 'valid'` metadata in
  `train_val_atlas.csv`;
- when `MODE = 'valid'`, forces `PART_ID = 0` and `TOTAL_PARTS = 1`;
- writes validation files as `ubc_ocean_valid.bin` and `ubc_ocean_valid.csv`;
- uses the same binary format as the other shard generator:
  64-byte header, `UBC_DATA` magic, CHW layout, `uint8`, and `3x256x256`
  patches;
- contains a separate merge/shuffle path for training shards that reads
  `./shards/*_part_*.csv/bin`, performs a global
  `sample(frac=1, random_state=42)`, writes `ubc_train_shuffled.bin` and
  `ubc_train_shuffled.csv`, recomputes CRC32, and prints a run-length entropy
  report.

The committed copy of this notebook has `execution_count = null` and no outputs
for all cells. The merge/shuffle cell is therefore historical/local-script
evidence rather than proof of a completed Kaggle run. It also assumes local
helper state such as `glob` and the `./shards` directory. The uploaded Kaggle
dataset filename and file sizes, plus user confirmation, are the current source
of truth that the training shard is already correctly pre-shuffled.

As committed, the notebook still uses `MODE = 'train'` or `MODE = 'valid'`,
writes `split = 'train'` and `split = 'valid'`, and does not yet write
`split = 'test'` or `ubc_ocean_test.*` files. Treat it as a starting point for a
test-set generator, not as proof that the test set already exists. The test
generator should explicitly invert the train/valid exclusion and operate on the
masked-WSI pool.

The autoencoder and later supervised classifier can share the same sealed test
set once generated. Spec 0001 should make both validation and test sources
explicit: validation can come from the confirmed pre-shuffled dataset, while test
must be generated, uploaded, and kept out of the tuning loop.

## Carry Forward

Carry these ideas forward into spec 0001 unless a later decision changes them:

- start with 256x256 RGB patches;
- keep `[-1, 1]` normalization unless the steerable path requires a change;
- keep a configurable HED corruption policy shared by both comparison branches;
- keep fixed 25-patch qualitative artifacts;
- keep the confirmed pre-shuffled train/validation shards explicit in configs;
- generate and seal a held-out test shard from the masked-WSI pool before final
  evaluation or paper claims;
- keep checkpoint/resume support;
- keep Kaggle DDP/AMP/compile as isolated launch/runtime concerns;
- keep exact dataset slugs in metadata, not UI display names;
- keep dataset roots configurable and verified at launcher start.

## Do Not Carry Forward Blindly

Do not copy these FSQ-era behaviors into the new first baseline:

- FSQ, vector quantization, codebooks, discrete indices, or bin histograms;
- PixelShuffle/sub-pixel upsampling;
- 1x1 pointwise projections in the comparable first-run architecture;
- Charbonnier loss or FSQ-era objective terms;
- FSQ excursion penalties;
- resume from `maximusshtefan/non-eq-vae-output`;
- only 90-degree `rot90` checks as the equivariance evaluation;
- metrics without sample count `n`;
- notebook-generated source code as the canonical implementation.

## Spec 0001 Reopened Decisions

`docs/specs/0001-translatable-normal-vae-baseline.md` was reopened on
2026-06-11 after user correction and adversarial review. It is not currently
implementation-ready.

Current reopened direction:

- first-run input size: 256x256 RGB;
- first-run latent shape: `(B, 16, 32, 32)`, matching the useful spatial
  bottleneck scale of the historical FSQ run while removing FSQ quantization and
  the learned bottleneck scale `s`;
- first-run macro-architecture: keep the broad historical FSQ/ResNet18-like
  residual encoder-decoder shape, but replace FSQ, PixelShuffle, 1x1
  projections, and other non-translatable details. Keep standard GroupNorm in
  the Conv2d baseline for real-run stability, while replacing raw GroupNorm with
  field-aware normalization in the SO(2) path. Projection shortcuts should follow
  a ResNet-D/anti-aliased-style policy: branch-local fixed fieldwise
  downsample/upscale primitives replace learned stride in stage-transition
  branches, with 5x5 projection convolution for channel changes, not naive
  pointwise or one-shot shape adapters. The exact downsample operator must be
  selected from the future `SO(2)` side first and then mirrored exactly in the
  non-equivariant baseline;
- first-run normalization policy: baseline uses `torch.nn.GroupNorm(8, C,
  affine=True)` in hidden blocks; SO(2) uses repo-owned field-aware norm with
  scalar affine bias and no additive vector bias;
- first-run activation policy: full-mixing scalar Conv2d baseline channels and
  future `SO(2)` scalar/trivial fields use the same learned pointwise scalar
  gate; future `SO(2)` nontrivial irrep fields use learned radial gates with no
  additive vector bias; activation gates use learned `a,b` and no `gamma`;
- first-run output policy: no final `tanh`; use a zero-initialized final RGB
  convolution, train L1 on raw normalized output, and clamp/project only for
  SSIM, PSNR, saved images, and artifacts outside the model forward path;
- first-run corruption: corrected Tellez-style HED/OD stain jitter plus
  per-image Gaussian noise sampled from `Uniform(0.0, 0.05)`;
- first-run objective: `L1 + 0.1 * (1 - SSIM) + beta * KL`, with repo-owned
  FP32 Torch SSIM suitable for compiled/offline Kaggle execution;
- beta warmup: first epoch for epoch-based runs, 10 percent of optimizer steps
  for step-limited debug runs, no cyclic restarts;
- first real baseline run: 10 epochs, with validation and checkpointing every
  half epoch;
- checkpoint retention: keep `best_model.pt`, the final checkpoint, and the
  latest four interval checkpoints, preserving the useful FSQ retention idea
  without reusing FSQ checkpoint formats;
- Active Spec 0011 v4 ranks complete dual-T4 recipe/batch pairs by
  `floor(real_train_patch_count/global_batch) * synchronized mean steady-step wall
  time`. It has no minimal-toggle or total GPU-time objective: inventory all plausible
  installed acceleration values including experimental ones, prove exclusions, begin
  from maximal compatible bundles, and test sealed complex interactions directly;
- the 309 recovered v2 probes are immutable canonical repo evidence, not a resume prefix
  or standing. They must prevent repeated old singleton/all-batch sweeps and guide batch/
  recipe order; only exact identities close roles, and old `p00310` never executes;
- the current partial v3 implementation is quarantined fail-closed work. V4 removes the
  obsolete 118-slot/two-session cap, uses deterministic finite action closure across as
  many verified resumable sessions as necessary, and still requires relock, local
  preflight, explicit remote approval, and independent verification before execution;
- compute narrowing alone cannot select the runtime. The later generated-file and real-
  data gates must still measure loader starvation and paired reconstruction quality/LR
  before writing `selected_runtime.json`;
- the short benchmark/debug path must log gate-health telemetry for learned gate
  `a,b` parameters so non-finite values, persistent saturation, or hidden-block
  collapse are caught before full training.

Current 2026-06-12 local slice status:

- the topology-count artifact now verifies the recorded analytic Conv2d
  baseline count target for the reopened `32x32x16` ResNet-like residual
  architecture schedule, including projection shortcuts and fixed resampling
  operators. See spec 0001 for the `topology_count_ready` exception and current
  artifact schema;
- the branch-local non-naive ResNet-D/anti-aliased-style residual
  projection/downsample operator remains locked to the repo-owned 5x5 separable
  binomial low-pass + decimation operator unless a later SO(2) spike supersedes
  it;
- the narrow local `data_metrics_ready` slice now implements UBC-format
  synthetic patch shards, exact header/CRC parsing, split validation with
  `synthetic_pass` versus real `pass|warn|fail`, and repo-owned FP32
  MAE/MSE/PSNR/full-SSIM metrics. This is local evidence only, not a Kaggle
  runtime or paper-claim unlock;
- the narrow local `fixed_selectors_dataloader_ready` slice now implements
  deterministic data-root resolution, read-only mmap tensor-only loading, and
  fixed selector schema/generation/validation with synthetic tests. Real fixed
  selector generation remains a local data-access step against the real Kaggle
  train/validation shards and is not a Kaggle remote execution step;

Remaining implementation-relock blockers:

- enforce the verified Kaggle T4 benchmark metadata value
  `machine_shape = "NvidiaTeslaT4"` and the safe single-visible-GPU versus
  dual-DDP launch mode before remote benchmark push; `dual_t4_ddp` rows must
  prove two visible T4 devices and two ranks at runtime;
- run clean-context adversarial spec review after the count target, metadata,
  and quality/import routes are integrated.

Full-run blockers after implementation:

- complete and strictly verify the resumable Spec 0011 v4 baseline dual-T4 compute
  selector; unfinished scope, corrupt lineage, or verifier failure means incomplete with
  no narrowing;
- run the generated-file then real-data loader, paired quality, and LR gates for the
  verified compute finalists before selecting per-device/global batch and runtime;
- require dataloader throughput and paired numerical-check artifacts for the
  selected row;
- confirm the gate-health summary has no non-finite gates, persistent near-total
  saturation, or unexplained hidden-block collapse;
- run selected-runtime real-data debug, checkpoint/resume, and tiny-overfit
  checks before the first 10-epoch baseline run;
- write the selected runtime to the benchmark artifacts and resolved full-run
  config before the first 10-epoch baseline run.

Remaining final-claim blockers:

- generate, upload, and lock the sealed masked-WSI test shard from
  `docs/data/ubc_ocean_masked_holdout_ids.csv`;
- verify the final Kaggle test dataset slug, mount path, manifest, and provenance
  before using any result for paper claims.
