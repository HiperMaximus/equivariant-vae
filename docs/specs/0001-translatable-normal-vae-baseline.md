# Spec 0001: Translatable Normal VAE Baseline

Status: draft active / reopened for architecture and objective corrections
Implementation readiness: not locked
Owner/workstream: comparable non-equivariant VAE baseline
Last updated: 2026-06-11

## Purpose

Replace the historical FSQ autoencoder experiment with a normal denoising VAE
baseline whose operations can be translated to the future continuous `SO(2)`
steerable model.

This is the first implementation target before building the full `escnn` path.
The spec was reopened after adversarial review and user correction: the previous
`4x4` latent target was too compressed for the intended spatial-coherence
comparison. This spec must be relocked only after the open questions near the end
are resolved. It is not a final paper-claim contract: final claims still require
a sealed masked-WSI test shard.

## Non-Goals

- No FSQ, vector quantization, codebooks, discrete indices, or quantized latent
  telemetry.
- No FSQ bottleneck scalar scale parameter `s`, straight-through rounding, or
  tanh-bounded latent-domain trick. The Gaussian VAE bottleneck uses only `mu`,
  `logvar`, sampled `z`, and the declared KL policy.
- No final performance claims against the steerable model.
- No thesis repo update.
- No reflection, `O(2)`, or other symmetry ablation.
- No arbitrary baseline layer unless its steerable counterpart is already known
  or explicitly documented as a temporary non-comparable ablation.
- No Kaggle push until the placeholder kernel is replaced and the user gives
  explicit push permission.

## Inputs And Data Contract

First-run input contract:

- dataset: UBC-OCEAN histopathology patches from the current Kaggle pipeline;
- source behavior inventory: `docs/behavior_inventory_kaggle.md`, derived from
  `kaggle/train_runs`, `kaggle/dataset_generation`, and
  `kaggle/generate_dataset_Classification_With_Masks`;
- image size: 256x256 RGB;
- binary patch shape: `3x256x256`, CHW, `uint8`, 64-byte `UBC_DATA` header;
- normalization: convert `uint8` to float in `[-1, 1]` with
  `x = image.float() / 127.5 - 1.0`;
- model output range: `tanh`, also in `[-1, 1]`;
- training input: corrupted patch `x_in = corrupt(x_clean)`;
- target: clean patch `x_clean`;
- train/validation source: confirmed Kaggle dataset
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`;
- required train files: `dataset/ubc_train_shuffled.bin` and
  `dataset/ubc_train_shuffled.csv`;
- required validation files: `dataset/ubc_ocean_valid.bin` and
  `dataset/ubc_ocean_valid.csv`;
- CSV schema rule: load by column name; `idx` is optional because train metadata
  has `wsi_id,label,x,y` while validation metadata has `idx,wsi_id,label,x,y`;
- train/validation split verification: 322 train WSIs and 39 validation WSIs,
  both non-TMA and with zero overlap with supplemental-mask image IDs;
- patch CSV label mapping: `0=CC`, `1=EC`, `2=HGSC`, `3=LGSC`, `4=MC`;
- train/validation patch counts: 300000 train patches and 30000 validation
  patches;
- local tests and debug runs must support synthetic/generated patch shards so
  the laptop workflow does not require downloading the Kaggle binaries.

Masked holdout contract:

- the pre-shuffled patch dataset does not contain a held-out test shard;
- exact masked holdout candidate list:
  `docs/data/ubc_ocean_masked_holdout_ids.csv`;
- candidate pool: 152 UBC-OCEAN non-TMA WSIs with supplemental masks;
- mask policy: supplemental masks are not exhaustive over each WSI; use masked
  WSIs as a held-out slide pool, but do not treat unmasked regions inside those
  WSIs as exhaustive negative labels;
- target sealed test dataset slug to create:
  `maximusshtefan/ubc-ocean-masked-holdout-patches`;
- target sealed test files:
  `dataset/ubc_ocean_test.bin`, `dataset/ubc_ocean_test.csv`,
  `dataset/ubc_ocean_test_manifest.csv`, and
  `dataset/ubc_ocean_test_provenance.json`;
- until that dataset exists and is locked, runs are train/validation-only and
  must not be used for final paper claims.

## Corruption Contract

Use the same denoising corruption policy for the baseline and future steerable
model:

- apply corruption to `x_clean` after `[-1, 1]` normalization;
- use a Tellez-style HED/optical-density stain jitter plus mild image-space
  Gaussian noise as the first implementation;
- cite and frame this as stain-domain randomization for robust denoising, not as
  a calibrated physical scanner or section-thickness simulator;
- do not copy the historical notebook corruptor from `kaggle/train_runs`; a
  clean implementation is required because the historical CHW/HED matrix
  convention is ambiguous and likely wrong for channel-first left multiplication;
- default config:
  - `corrupt_prob = 0.30`;
  - H/E alpha range `[0.80, 1.20]`;
  - H/E beta range `[-0.05, 0.05]`;
  - D alpha range `[0.98, 1.02]`;
  - D beta range `[-0.01, 0.01]`;
  - image-space Gaussian noise standard deviation sampled per image from
    `Uniform(0.0, 0.05)` on the `[-1, 1]` image scale, so the denoiser cannot
    memorize one fixed noise variance;
- log whether an image was corrupted and the RNG seed policy for reproducible
  debug runs.

Required stain-corruptor implementation rules:

- store the fixed HED stain matrix with an explicit channel-first convention;
- add unit tests against a known HED/RGB convention such as `scikit-image`
  `rgb2hed`/`hed2rgb` on small tensors, or include a documented copied matrix
  convention if `scikit-image` is not a runtime dependency;
- identity parameters must round-trip RGB within tolerance before noise;
- H/E and D perturbation parameters must affect the intended HED channels, not
  transposed mixtures;
- use explicit Torch RNG state derived from `corruption_seed`, rank, sample
  identity, and optimizer step where applicable;
- do not consume stain/noise RNG in clean validation mode;
- preserve input shape and dtype, and document any memory-format conversion;
- generate a fixed 25-patch visual QA artifact showing clean, stain-corrupted,
  Gaussian-only, and combined corrupted patches before the first Kaggle baseline
  run.

Relevant stain-domain references to cite in the paper/spec implementation:

- Ruifrok and Johnston, 2001, optical-density color deconvolution:
  https://pubmed.ncbi.nlm.nih.gov/11531144/;
- Macenko et al., 2009, OD/SVD stain normalization:
  https://doi.org/10.1109/ISBI.2009.5193250;
- Vahadane et al., 2016, structure-preserving stain normalization:
  https://doi.org/10.1109/TMI.2016.2529665;
- Tellez et al., 2018/2019, HED stain augmentation and HED-light/HED-strong
  benchmark settings: https://arxiv.org/abs/1808.05896 and
  https://arxiv.org/abs/1902.06543;
- RandStainNA, 2022, data-driven stain augmentation in HED/HSV/LAB spaces:
  https://arxiv.org/abs/2206.12694.

## Model Contract

The baseline must be generated from a layer schedule that the equivariant model
can reuse. The non-equivariant convolutions are ordinary `torch.nn.Conv2d`; all
channels are treated as scalar tensor channels, and each convolution may freely
mix all input channels. The macro-topology, capacity bookkeeping, kernels,
upsampling, latent shape, and gate family must mirror the planned `SO(2)` path.

First-run fixed choices:

| Item | Value |
| --- | --- |
| Input | 256x256 RGB |
| Latent | spatial Gaussian latent `(B, 16, 32, 32)` |
| Normalization layers | none |
| Stem kernel | 7x7, same padding |
| Hidden/down/up kernels | 5x5, same padding |
| VAE head kernels | 5x5, same padding |
| Upsampling | bilinear scale factor 2 followed by convolution |
| Output | final 5x5 convolution to RGB plus `tanh` |
| KL convention | mean over batch, latent channels, and latent spatial positions |

Encoder spatial schedule:

```text
256 -> 128 -> 64 -> 32
```

Channel and future field-capacity schedule:

| Spatial size | Total channels | Future scalar fields | Future F1 irrep copies | Future F2 irrep copies |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 16 | 8 | 0 |
| 128 | 48 | 16 | 12 | 4 |
| 64 | 64 | 24 | 12 | 8 |
| 32 | 96 | 32 | 20 | 12 |
| latent | 16 | 16 | 0 | 0 |

For the first non-equivariant baseline, the table is capacity bookkeeping only:
it does not restrict Conv2d mixing, activation grouping, or residual addition.
The scalar/F1/F2 columns define the planned `SO(2)` field multiplicities that a
future steerable model should mirror when reporting capacity and parameter/FLOP
differences.

The first complete run uses scalar/trivial latent fields only. Frequency-1 or
frequency-2 latent fields require a follow-up spec because `logvar`, sampling,
and KL must become representation-aware.

Encoder block pattern:

```text
Stem:
  Conv2d(3 -> 32, kernel=7, stride=1, padding=3)
  ActivationPolicy

Down block i:
  Conv2d(C_i -> C_{i+1}, kernel=5, stride=2, padding=2)
  ActivationPolicy
  Conv2d(C_{i+1} -> C_{i+1}, kernel=5, stride=1, padding=2)
  ActivationPolicy

VAE heads:
  mu_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2)
  logvar_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2)
```

Decoder mirrors the encoder:

```text
Latent projection:
  Conv2d(16 -> 96, kernel=5, stride=1, padding=2)
  ActivationPolicy

Up block i:
  bilinear upsample(scale_factor=2, align_corners=False)
  Conv2d(C_i -> C_{i-1}, kernel=5, stride=1, padding=2)
  ActivationPolicy
  Conv2d(C_{i-1} -> C_{i-1}, kernel=5, stride=1, padding=2)
  ActivationPolicy

Output:
  Conv2d(32 -> 3, kernel=5, stride=1, padding=2)
  tanh
```

Allowed first-run operations:

- odd square `Conv2d` with 5x5 or 7x7 kernels;
- strided odd-kernel convolution for downsampling;
- bilinear upsampling plus convolution for upsampling;
- spatial Gaussian VAE latent map;
- scalar gated activation policy defined below.

Residual policy:

- first-run residual/ReZero/Fixup connections are disabled;
- do not add residual branches, learned residual scales, or skip projections in
  spec 0001 implementation;
- if residuals are needed later for stability, write a follow-up spec with exact
  locations, parameters, initialization, parameter/FLOP impact, and `SO(2)`
  counterpart.

Banned first-run operations:

- FSQ or any vector-quantized bottleneck;
- PixelShuffle or sub-pixel convolution;
- nearest-neighbor upsampling in the comparable path;
- 1x1 pointwise convolutions;
- depthwise/grouped/MBConv/squeeze-excite/channel-attention operations;
- raw `GroupNorm`, `BatchNorm2d`, `LayerNorm`, per-channel affine normalization,
  or channel dropout;
- arbitrary flattening, channel slicing, `.chunk()`, or tensor reshaping that
  cannot be mapped to future `GeometricTensor` field boundaries;
- FSQ-era resume sources or discrete-latent artifact requirements.

## Activation Contract

Do not use arbitrary componentwise SiLU everywhere. The baseline must use a
shared gated activation family that the future `SO(2)` model can mirror.
SiLU/Swish is treated as `x * sigmoid(x)`, not as a special unrelated
nonlinearity.

For the non-equivariant baseline, every hidden channel is a scalar tensor
channel. Apply the scalar gate componentwise to all channels, and allow the
surrounding Conv2d layers to mix channels freely.

Baseline scalar gate:

```text
gate_i = sigmoid(alpha_i * x_i + beta_i)
out_i = gamma_i * gate_i * x_i
```

Rules for the baseline scalar gate:

- `alpha_i`, `beta_i`, and `gamma_i` are learned scalar parameters per channel;
- initialize to ordinary SiLU/Swish behavior where possible
  (`alpha=1`, `beta=0`, `gamma=1`);
- do not add extra activation scalars only to the future equivariant model to
  compensate for parameter count differences;
- do not tie or group baseline activation parameters by future field schedule in
  the first run. Any grouped activation tying requires a later explicit
  ablation/spec.

Future `SO(2)` counterpart:

- scalar/trivial fields use the same scalar gate family;
- nontrivial 2D irrep copies use a radial gate over an invariant norm;
- this radial gate is implemented and tested as part of the activation policy,
  but it is not applied to fake vector pairs in the first scalar Conv2d
  baseline.

For each future 2-channel irrep copy `v = (u, w)`:

```text
r2 = u**2 + w**2
gate = sigmoid(a_i * r2 + b_i)
out = gamma_i * gate * v
```

Rules for future radial gates:

- the two components in a vector pair must share the same gate;
- initialize future vector/irrep copies near pass-through with a mildly positive
  gate bias and stable `gamma`; document the exact initialization in config/code;
- future vector/irrep copies may have scalar gate bias `b_i`, but must not have
  an additive learned 2D vector bias because that would break `SO(2)`
  equivariance;
- implement this as an explicit `GatedScalarActivation`,
  `RadialGate`, and `ActivationPolicy` module using a central field schedule,
  not ad hoc reshaping inside model blocks;
- add a unit test that rotates synthetic vector pairs and verifies
  `activation(rho(theta) v) == rho(theta) activation(v)` within tolerance.

No normalization layers are allowed in the first implementation. If training
stability later requires normalization, write a follow-up spec and prove/test the
equivariant counterpart first.

## Objective Contract

Use a normal denoising VAE with a composite reconstruction objective:

```text
z = mu + exp(0.5 * logvar) * eps
l1_loss = mean(abs(x_hat - x_clean))
ssim_loss = 1 - ssim(x_hat, x_clean)
recon_loss = l1_loss + ssim_weight * ssim_loss
kl_element = -0.5 * (1 + logvar - mu ** 2 - exp(logvar))
kl_loss = mean(kl_element)
loss = recon_loss + beta * kl_loss
```

This objective is a composite beta-VAE-style objective, not a strict Gaussian
ELBO. Keep MSE and PSNR as metrics, but do not optimize MSE in the first run.
Implement SSIM as repo-owned Torch code that runs in FP32 and can be included in
the compiled step function without internet or undeclared Kaggle dependencies.
First locked `ssim_weight`: `0.1`.

First beta policy:

- full epoch-based runs: linear warmup from 0 to 1 over the first full epoch,
  then keep beta fixed at 1;
- tiny step-based debug runs: linear warmup from 0 to 1 over the first 10 percent
  of configured optimizer steps;
- no cyclic beta restarts in the first locked run;
- beta value must be logged per optimizer step.

AMP and GradScaler policy:

- support both AMP and non-AMP execution; do not assume AMP is faster until the
  Kaggle runtime benchmark measures it;
- if AMP `GradScaler` detects non-finite gradients and skips
  `optimizer.step()`, call `scaler.update()`, zero gradients, log
  `amp_step_skipped = 1`, and continue to the next batch;
- do not retry the same batch after a skipped AMP step;
- do not increment `global_step`, advance LR or beta schedulers, run
  step-triggered validation/checkpointing, or count the batch as an optimizer
  update when the optimizer step was skipped.

Log at minimum:

- total loss;
- reconstruction loss;
- L1 loss;
- SSIM loss and SSIM metric;
- KL loss;
- beta;
- SSIM, MAE, MSE, PSNR;
- posterior `mu` mean/std/min/max;
- posterior `logvar` mean/std/min/max/clamp count;
- learning rate;
- `event_id`, `batch_attempt`, `optimizer_step`, and `amp_step_skipped`;
- sample count `n` for every metric summary.

MSE is a metric for the first run, not a training loss.

Validation/evaluation modes:

- `eval_clean`: encoder input is `x_clean`, target is `x_clean`; do not call the
  corruptor or consume corruption RNG;
- `eval_corrupted`: encoder input is `corrupt(x_clean)` with fixed validation
  corruption seed and logged corruption config, target is `x_clean`;
- report reconstruction metrics and KL terms for both modes with separate
  `split` and `view` labels, never as one pooled number. Required `view` values
  are `eval_clean` and `eval_corrupted`.

## Training And Config Contract

All values that affect the experiment must live in YAML configs, not hidden
inside model or CLI code. CLI flags may override config values only when the
override is recorded in the run config snapshot.

Required seed policy:

- `global_seed = 20260610`;
- `data_seed = 20260610`;
- `corruption_seed = 20260611`;
- `latent_seed = 20260612`;
- save Python, NumPy, and Torch RNG state in checkpoints.

Required optimizer and schedule defaults:

| Field | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `1e-5` |
| Gradient clipping | global norm `1.0` |
| LR warmup | linear warmup over first 5 percent of configured train steps |
| LR schedule | cosine decay to `5e-6`, no restarts |
| Beta warmup | first epoch for epoch-based runs; first 10 percent of optimizer steps for step-limited debug runs; no cyclic restarts |
| `logvar` clamp | clamp to `[-8.0, 4.0]` before sampling and KL |

Runtime benchmark requirement before the first full Kaggle run:

- the benchmark is a short decision run, not training; it must finish in minutes
  rather than multiple hours and must stop after fixed warmup/measured steps;
- use the real data loader and training step, but do not run a full epoch
  schedule or tune model quality during the benchmark;
- benchmark single T4 and dual T4 DDP;
- for each GPU configuration, benchmark AMP off/on and `torch.compile` off/on
  where the runtime supports it;
- for each row, record warm steady-state samples/sec, step time, compile
  overhead, max VRAM, largest stable per-device batch, global batch,
  `amp_step_skipped` count, and any compile/DDP failure;
- batch size is selected from VRAM and throughput evidence for each runtime
  configuration, not hard-coded from the historical FSQ run.

The selected baseline runtime must be recorded in the resolved config. Use
`per_device_batch_size`, `global_batch_size`, `mixed_precision.enabled`, and
`torch_compile.enabled`; do not leave the batch-size or precision meaning
ambiguous.

Required first-run budget defaults:

| Config | Batch size | Train steps | Validation interval | Checkpoint interval |
| --- | ---: | ---: | ---: | ---: |
| `non_eq_vae_debug_cpu.yaml` | 2 global | 8 | 4 | 4 |
| `non_eq_vae_kaggle_runtime_benchmark.yaml` | searched per device | short fixed benchmark steps | optional one fixed validation micro-pass | none except benchmark summary |
| `non_eq_vae_kaggle_debug.yaml` | benchmarked per device | 200 | 50 | half epoch or 100 steps |
| `non_eq_vae_baseline.yaml` | benchmark-selected per device | 10 epochs | half epoch | half epoch |

The future `SO(2)` model must use the same training budget and validation
access, unless a later run spec explicitly supersedes both models together.

Checkpoint retention:

- epoch-based runs save and validate every half epoch;
- retain `best_model.pt`, the final checkpoint, and the latest four interval
  checkpoints, mirroring the useful FSQ-era retention behavior without reusing
  FSQ checkpoint formats;
- record checkpoint pruning decisions in the run manifest;
- resume must restore model, optimizer, LR scheduler, beta scheduler, AMP scaler
  when present, epoch/progress counters, config hash, and RNG state.

Required output schemas:

- `config_resolved.yaml`: full config after CLI overrides;
- `metrics/train_steps.csv`: one row per logged train step with at least
  `run_name,event_id,batch_attempt,optimizer_step,split,loss,recon_loss,l1_loss,ssim_loss,kl_loss,beta,lr,grad_norm,batch_size,amp_step_skipped`;
- skipped AMP rows are logged as batch-attempt events with
  `amp_step_skipped = 1`; they do not increment `optimizer_step` and do not
  trigger optimizer-step-based schedules, validation, or checkpointing;
- `metrics/validation_steps.csv`: one row per validation event with at least
  `run_name,optimizer_step,split,view,n,mse_mean,mae_mean,psnr_mean,ssim_mean,kl_mean`;
- `eval/per_image_metrics.csv`: one row per evaluated patch with at least
  `sample_id,split,view,wsi_id,label,x,y,mse,mae,psnr,ssim`;
- `eval/summary.json`: mean, standard deviation, and `n` for every metric,
  grouped by `split` and `view`;
- `artifacts/manifest.json`: paths and provenance for every generated figure;
- `benchmark/runtime_matrix.csv`: one row per benchmarked runtime configuration;
- `benchmark/selected_runtime.yaml`: selected accelerator, compile, AMP, and
  batch-size decision for the first full run;
- `checkpoints/step_*.pt`: model, optimizer, scheduler, beta scheduler, scaler
  if present, current step, config hash, and RNG state.

`logvar_clamp_count` must be logged whenever any values are clamped.

## Fixed 25-Patch Protocol

The qualitative 25-patch set must be deterministic and shared by the baseline
and future `SO(2)` model.

Selection policy for `configs/spec0001/fixed_25_validation_patches.yaml`:

1. Use the validation CSV from `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
2. Group rows by numeric label `0..4`.
3. For each row, compute
   `sha256("20260610:{wsi_id}:{label}:{x}:{y}")`.
4. Sort each label group by that digest, then by `wsi_id,x,y`.
5. Select the first 5 rows per label.
6. Store the ordered 25 selectors with `wsi_id,label,x,y,source_split`.

The artifact command may accept `--fixed-count 25`, but implementation must fail
if the fixed-patch config is missing or if the selected count is not exactly 25.
Do not silently resample a different set.

Because the Kaggle validation CSV is not committed in this repo, the first
implementation must include a deterministic selector generator:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.select_fixed_patches \
  --config configs/spec0001/non_eq_vae_kaggle_debug.yaml \
  --data-root auto \
  --output configs/spec0001/fixed_25_validation_patches.yaml
```

This generator requires access to the real validation CSV and is therefore a
data-access step, not a pure offline local test. Local synthetic tests may use a
separate generated synthetic selector under `runs/` but must never overwrite the
canonical fixed-25 config.

## Rotated And Latent Artifact Protocol

Spec 0001 must produce baseline-compatible placeholders or outputs for the
advisor-requested rotated-input and latent visualizations without pretending the
scalar baseline has nontrivial `SO(2)` latent fields.

Baseline protocol:

- use the deterministic posterior mean `mu`, not sampled `z`, for latent
  transformation artifacts;
- rotate image inputs with the documented interpolation, padding/cropping, and
  boundary-mask policy used by the evaluator;
- for the scalar spatial latent baseline, the transformed-latent path is:
  encode clean input to `mu`, spatially rotate the 16-channel latent map with the
  same angle convention, decode with the decoder path, then compare against the
  rotated-input reconstruction;
- report boundary-masked and unmasked error maps;
- store angle list, interpolation mode, padding mode, align-corners policy, and
  mask policy in `artifacts/manifest.json`;
- future nontrivial `SO(2)` latent-field transformations require a follow-up
  spec because `mu`, `logvar`, sampling, KL, and representation action become
  irrep-aware.

## Required Implementation Artifacts

Expected package root:

```text
src/eqvae/
```

Required modules:

- `src/eqvae/data/patch_shards.py`: UBC binary/CSV patch shard dataset;
- `src/eqvae/data/synthetic.py`: tiny deterministic synthetic patch shards for
  local tests and smoke runs;
- `src/eqvae/data/splits.py`: WSI and masked-holdout split validation helpers;
- `src/eqvae/corruption/stain.py`: Tellez-style HED/OD stain jitter and
  Gaussian noise corruption with corrected matrix convention;
- `src/eqvae/models/field_schedule.py`: tensor-channel schedule and future
  `SO(2)` field multiplicity metadata;
- `src/eqvae/models/activations.py`: gated scalar activation and future radial
  gate policy;
- `src/eqvae/models/non_equivariant_vae.py`: translatable Conv2d VAE factory;
- `src/eqvae/losses/vae.py`: reconstruction, KL, and beta schedule;
- `src/eqvae/metrics/reconstruction.py`: SSIM, MAE, MSE, PSNR;
- `src/eqvae/artifacts/`: boxplots, dashboards, fixed-patch grids, rotated-input
  grids, rotated-input versus latent grids, and latent visualization helpers;
- `src/eqvae/checkpointing.py`: save/resume with RNG state;
- `src/eqvae/cli/`: `smoke`, `train`, `benchmark_runtime`,
  `select_fixed_patches`, `evaluate`, and `artifacts` entry points.

Required config files:

- `configs/spec0001/non_eq_vae_baseline.yaml`;
- `configs/spec0001/non_eq_vae_debug_cpu.yaml`;
- `configs/spec0001/non_eq_vae_kaggle_debug.yaml`;
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.yaml`;
- `configs/spec0001/ubc_ocean_masked_holdout_test.yaml`;
- `configs/spec0001/fixed_25_validation_patches.yaml`.

Run outputs should go under ignored `runs/` paths locally and `/kaggle/working`
on Kaggle.

Package/import policy:

- use `src/eqvae` as the implementation package root;
- local commands use `PYTHONPATH=src` until a packaging backend is explicitly
  added;
- Kaggle launchers must insert `payload/src` into `sys.path` before importing
  `eqvae`;
- adding a build backend or package-discovery metadata requires updating this
  spec and the lockfile.

Config and dependency policy:

- configs are YAML by contract, but spec 0001 is not lockable until the parser
  choice is recorded in `pyproject.toml`/`uv.lock` or the config format is
  changed to a standard-library format;
- repo-owned Torch SSIM must be implemented under `src/eqvae`; do not import
  `pytorch-msssim` in spec 0001 code unless a later spec deliberately changes
  the offline/compiled SSIM policy;
- if `pytorch-msssim` remains in `pyproject.toml` as historical dependency debt,
  record it in the quality/dependency cleanup route before locking spec 0001.

Local CPU smoke policy:

- CPU smoke tests are shape/contract tests, not speed benchmarks;
- CPU `torch.compile` tests may use tiny synthetic batches and must have bounded
  step counts so they do not turn into long local training jobs;
- CPU float16 smoke is allowed to be a narrow dtype-path check with documented
  tolerances or explicit skips for unsupported CPU operations;
- GPU speed, AMP, and DDP behavior are decided only by the permission-gated
  Kaggle runtime benchmark.

Implementation milestones before broad coding:

1. Spec relock slice: parameter/FLOP counting, residual-off confirmation,
   package/import policy, config parser decision, quality-debt route, fixed-25
   selector plan, artifact protocol, and final clean-context spec review.
2. Skeleton slice: `src/eqvae`, `configs/spec0001`, no-sync import smoke, and
   one CPU pytest proving CLI/import wiring.
3. Data/metrics slice: patch-shard loader, synthetic data, split validation,
   MAE/MSE/PSNR/SSIM metrics.
4. Model/loss slice: activation policy, non-equivariant VAE shapes, KL/L1/SSIM
   loss, beta schedule, compile/precision smoke.
5. Corruption slice: tested HED/OD stain jitter, Gaussian corruption, RNG policy,
   and 25-patch visual QA.
6. Train/resume slice: optimizer/scheduler, AMP skipped-step behavior,
   checkpoint save/resume, metrics schemas, retention.
7. Artifact/evaluation slice: fixed-25 selector, evaluator, boxplots,
   dashboards, rotated/latent artifacts.
8. Kaggle slice: payload build, debug launcher, local payload validation, then
   permission-gated remote benchmark/debug runs.

## Kaggle Packaging Contract

`kaggle kernels push` uploads only the kernel folder. Therefore the debug kernel
must be self-contained before it is push-ready.

Required generated kernel layout:

```text
kaggle/kernels/non_eq_vae_debug/
  kernel-metadata.json
  run.py
  payload/
    src/eqvae/
    configs/spec0001/
    pyproject.toml
    uv.lock
```

Required build command:

```bash
./scripts/kaggle_kernel.sh build
```

Build rules:

- copy only allowlisted implementation files into `payload/`;
- do not copy `.git`, `.venv`, paper files, historical notebooks, checkpoints,
  local run artifacts, credentials, or Overleaf data;
- `run.py` must insert `payload/src` into `sys.path` before importing `eqvae`;
- Kaggle internet stays disabled;
- first implementation must not require `pip install` or dependency resolution
  on Kaggle;
- metrics must use a repo-owned Torch SSIM implementation or another bundled
  implementation, not an undeclared network dependency;
- if a future dependency is unavailable on Kaggle, bundle a wheel under
  `payload/wheels/` and install with `--no-index --find-links`, with a separate
  spec update first.

Kaggle debug metadata must keep:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"],
"competition_sources": [],
"kernel_sources": [],
"model_sources": [],
"enable_internet": "false"
```

The historical FSQ output dataset `maximusshtefan/non-eq-vae-output` is forbidden
for spec 0001 kernels.

## Verification Commands

The implementation is accepted only when these exact local commands exist and
pass. Local commands must not create, sync, or refresh the environment. If the
repo-local `.venv` is missing or stale, ask the user before running:

```bash
uv sync --locked --python 3.12 --group dev
```

General repo checks:

```bash
./scripts/agent_preflight.sh
./scripts/python_quality.sh
```

Spec 0001 includes resolving or quarantining the historical strict-quality debt
in `main.py` / exploratory `src/nn` so `./scripts/python_quality.sh` passes.
Do not weaken global Ruff/BasedPyright settings and do not add global ignores.

Unit and contract tests:

```bash
PYTHONPATH=src uv run --locked --no-sync pytest \
  tests/test_patch_shards.py \
  tests/test_stain_corruptor.py \
  tests/test_activation_policy.py \
  tests/test_translatable_vae_shapes.py \
  tests/test_vae_loss.py \
  tests/test_metrics_artifacts.py \
  tests/test_banned_operations.py
```

Torch compile and precision smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync pytest tests/test_compile_precision_smoke.py
```

Local CPU synthetic smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.smoke \
  --config configs/spec0001/non_eq_vae_debug_cpu.yaml \
  --data synthetic \
  --device cpu \
  --batch-size 2 \
  --compile inductor \
  --dtype float32
```

Local CPU float16 smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.smoke \
  --config configs/spec0001/non_eq_vae_debug_cpu.yaml \
  --data synthetic \
  --device cpu \
  --batch-size 1 \
  --max-steps 1 \
  --dtype float16
```

Debug train from scratch:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_debug_cpu.yaml \
  --data synthetic \
  --output-dir runs/local/spec0001-debug \
  --run-name spec0001_cpu_debug \
  --max-train-steps 8 \
  --max-val-steps 2 \
  --save-every-steps 4
```

Resume from midpoint checkpoint:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_debug_cpu.yaml \
  --data synthetic \
  --resume runs/local/spec0001-debug/checkpoints/step_000004.pt \
  --output-dir runs/local/spec0001-resume \
  --run-name spec0001_cpu_resume \
  --max-train-steps 10 \
  --max-val-steps 2
```

Evaluator and summaries:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.evaluate \
  --config configs/spec0001/non_eq_vae_debug_cpu.yaml \
  --checkpoint runs/local/spec0001-debug/checkpoints/step_000008.pt \
  --data synthetic \
  --split validation \
  --max-samples 32 \
  --output-dir runs/local/spec0001-eval
```

Artifact generation:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.artifacts \
  --eval-dir runs/local/spec0001-eval \
  --fixed-count 25 \
  --angles=-90,-45,0,45,90 \
  --output-dir runs/local/spec0001-artifacts
```

Kaggle local scaffold checks:

```bash
./scripts/kaggle_kernel.sh build
./scripts/kaggle_kernel.sh validate
bash -n scripts/kaggle_kernel.sh
python3 -m json.tool kaggle/kernels/non_eq_vae_debug/kernel-metadata.json
```

Kaggle debug command that the script kernel must run after implementation:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_debug.yaml \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working \
  --run-name non_eq_vae_spec0001_kaggle_debug \
  --max-train-steps 200 \
  --max-val-steps 20 \
  --save-every-steps 100
```

Kaggle runtime benchmark command that the script kernel must run before the
first full run:

```bash
python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.yaml \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working/runtime_benchmark \
  --run-name non_eq_vae_spec0001_runtime_benchmark
```

Permission-gated remote check, not required for local implementation acceptance:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
```

Only run the remote push after all local commands pass and the user explicitly
approves the remote write.

## Local Implementation Acceptance Criteria

The local implementation is complete when:

1. all verification commands above pass;
2. model construction is generated from the locked layer/channel schedule;
3. banned-operation checks reject FSQ, PixelShuffle, nearest upsampling, 1x1
   convs, grouped/depthwise convs, attention blocks, and normalization layers;
4. the data loader validates binary header, shape, dtype, patch count, required
   CSV columns, optional `idx`, and train/validation WSI non-overlap;
5. the split validator checks exact train/validation patch counts, exact
   train/validation WSI counts, zero overlap with
   `docs/data/ubc_ocean_masked_holdout_ids.csv`, and non-TMA status whenever
   official `train.csv` metadata is available;
6. synthetic data tests do not require network, Kaggle, or GPU access;
7. stain-corruptor tests verify HED/RGB round-trip convention, per-channel
   perturbation semantics, fixed-seed reproducibility, DDP/rank seed separation,
   no RNG consumption in clean validation mode, finite outputs, and visual QA
   artifact generation;
8. CPU smoke tests instantiate data, model, corruption, loss, optimizer,
   evaluator, and artifact writers;
9. compile/precision smoke tests cover `torch.compile`, output shapes, and the
   configured float16 path without requiring a GPU;
10. debug training completes from scratch and writes metrics, config, checkpoint,
   and RNG state;
11. resume training restores checkpoint, optimizer, scheduler/beta state, and RNG
   state;
12. AMP skipped-step behavior is tested or exercised so skipped steps do not
    advance optimizer-step counters, LR/beta schedules, validation, or
    checkpoint cadence;
13. the runtime benchmark CLI exists, runs on a tiny local synthetic budget, and
    writes the expected schema without requiring GPU or network access;
14. checkpoint retention keeps `best_model.pt`, the final checkpoint, and the
    latest four interval checkpoints;
15. evaluator writes per-image SSIM, MAE, MSE, PSNR and summary mean/std/`n`
    separately for `eval_clean` and fixed-seed `eval_corrupted`;
16. artifact writer emits metric boxplots, dashboard, fixed 25-patch
    reconstructions, rotated-input grids, rotated-input versus latent grids, and
    latent visualization placeholders or outputs;
17. the fixed 25-patch config contains exactly 5 validation patches per label and
    all future qualitative commands read it rather than resampling;
18. Kaggle debug kernel runs bundled repo code through the CLI, not notebook
    source or a GitHub-linked notebook;
19. `scripts/kaggle_kernel.sh push` rejects wrong dataset slugs, historical FSQ
    output sources, internet-enabled metadata, missing payloads, and placeholder
    launchers;
20. runs without the sealed masked-WSI test shard are labeled
    train/validation-only and excluded from final paper claims;
21. `CURRENT.md`, `docs/specs/README.md`, and relevant workflow docs are updated
    with implementation status and verification results.

## Full Kaggle Run Acceptance Criteria

The first 10-epoch Kaggle baseline is not ready until:

1. local implementation acceptance passes;
2. the user explicitly approves the remote Kaggle write/run;
3. the short Kaggle runtime benchmark writes `benchmark/runtime_matrix.csv` and
   `benchmark/selected_runtime.yaml`, including AMP off/on and compile off/on
   evidence for single and dual T4;
4. the selected single/dual T4, per-device/global batch, AMP, and compile config
   is copied into the resolved full-run config;
5. the baseline run uses the selected runtime config, validates/checkpoints every
   half epoch, and keeps the declared checkpoint retention.

## Open Questions And Gates

Implementation-relock blockers:

1. Final channel/future-field schedule: is the current `32x32x16` scalar latent
   and 32/48/64/96 hidden schedule acceptable after parameter/FLOP counting?
2. Final clean-context adversarial spec review must pass after the edits and
   parameter/FLOP count are integrated.
3. Strict quality-debt route must be explicit: either clean the historical
   `main.py` / exploratory `src/nn` debt, move/quarantine it through an approved
   spec without weakening global strictness, or make spec 0002 define the exact
   accepted command boundary.
4. Config parser/dependency policy must be locked, including whether a YAML
   parser dependency is added or config format changes.
5. Package/import policy must be locked enough that the verification commands
   import `eqvae` without dependency sync.
6. Fixed-25 selector generation and baseline rotated/latent artifact semantics
   must remain exactly as specified above, or be revised before implementation.

Full-run blockers after implementation:

1. Runtime target: after the Kaggle benchmark matrix, should the full run use
   single GPU or dual T4 DDP, should AMP and/or `torch.compile` be enabled, and
   what is the selected per-device/global batch size?
2. The selected runtime must be written to `benchmark/selected_runtime.yaml` and
   the resolved baseline config before the first 10-epoch Kaggle run.

## Known Risks

- The first implementation may train less stably without normalization. Do not
  add normalization ad hoc; write a follow-up normalization spec if needed.
- The future radial gate can suppress vector/irrep copies if initialized poorly.
  Initialize it near pass-through and test gradient flow before the `SO(2)`
  model depends on it.
- CPU float16 behavior can differ from Kaggle GPU float16 behavior. Local smoke
  checks are a contract test, not a replacement for Kaggle debug training.
- The target sealed test slug may need to change before upload. If it changes,
  update this spec and all configs before making final claims.
- The baseline still receives ordinary Conv2d kernels, so fairness depends on
  keeping every other degree of freedom matched when the steerable model is
  implemented.

## Adversarial Checks Before Implementation PR Completion

- Does any operation violate the future continuous `SO(2)` translation path?
- Does the VAE objective accidentally omit KL or use the wrong reduction?
- Does corruption randomness differ between comparison branches?
- Do metric scripts include `n` and run on the same evaluation images?
- Do qualitative artifacts use the same fixed patch IDs for both future models?
- Does any split leak WSI, patient, site, or masked-holdout information?
- Does the baseline receive more tuning or validation access than the future
  equivariant model?
- Does the Kaggle launcher import repo code instead of embedding a notebook copy?

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/behavior_inventory_kaggle.md`
- `docs/equivariant_vae_transition_plan.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/decisions/0001-continuous-so2-scope.md`
- `docs/decisions/0002-normal-vae-baseline.md`
- `docs/specs/0002-strict-python-quality-gate.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
