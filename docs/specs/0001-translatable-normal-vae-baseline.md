# Spec 0001: Translatable Normal VAE Baseline

Status: draft active / reopened for architecture and objective corrections
Implementation readiness: not locked
Owner/workstream: comparable non-equivariant VAE baseline
Last updated: 2026-06-11

## Purpose

Replace the historical FSQ autoencoder experiment with a normal denoising VAE
baseline whose operations can be translated to the future continuous `SO(2)`
steerable model.

Keep the broad historical FSQ architecture family: the replacement baseline is
ResNet18-like, with residual basic blocks and a spatial encoder/decoder
topology. The replacement removes FSQ quantization and non-translatable
operations, not the ResNet-like macro-architecture.

This is the first implementation target before building the full repo-owned
continuous `SO(2)` path.
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
- model output: raw normalized RGB reconstruction values in the same coordinate
  system as the target, without a final `tanh` or hard output clamp;
- image-domain projections for SSIM, PSNR, saved images, and visual artifacts
  use `clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)`;
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

The baseline must be generated from a ResNet-like layer schedule that the
equivariant model can reuse. The non-equivariant convolutions are ordinary
`torch.nn.Conv2d`; all channels are treated as scalar tensor channels, and each
convolution may freely mix all input channels. The residual macro-topology,
capacity bookkeeping, kernels, upsampling, latent shape, and gate family must
mirror the planned `SO(2)` path.

First-run fixed choices:

| Item | Value |
| --- | --- |
| Input | 256x256 RGB |
| Latent | spatial Gaussian latent `(B, 16, 32, 32)` |
| Normalization layers | baseline `GroupNorm`; future SO(2) field-aware norm |
| Stem kernel | 7x7, same padding |
| Hidden/down/up kernels | 5x5, same padding |
| VAE head kernels | 5x5, same padding |
| Padding mode | zero padding for train/model code; border-cropped metrics for equivariance diagnostics |
| Upsampling | bilinear scale factor 2 followed by convolution |
| Future SO(2) kernel basis | Gaussian radial shells plus real angular harmonics, `L <= 2` |
| Output | zero-initialized final 5x5 convolution to raw RGB, no final `tanh` |
| KL convention | mean over batch, latent channels, and latent spatial positions |

Future `SO(2)` kernel-basis policy is locked for the first implementation:

- use repo-owned analytic polar-harmonic basis construction with Gaussian radial
  shells and real angular harmonics `cos(m theta), sin(m theta)`;
- 5x5 kernels use radial shell centers `[0, 1, 2]`;
- 7x7 kernels use radial shell centers `[0, 1, 2, 3]`;
- use approximate ring widths `0.6` for interior rings, `0.4` for the outer
  ring, and a tiny origin width;
- angular frequencies `m > 0` have zero support at the kernel center; the center
  sample may only carry the `m = 0` spatial angular component, while still
  allowing legal intertwiners between compatible same-frequency input and output
  irreps;
- precompute basis buffers and learn only expansion coefficients;
- expand to dense `conv2d` inside the compiled forward path;
- allow scalar-output bias only where the representation policy permits it;
- keep Fourier-Bessel/Bessel bases as a future fallback/ablation requiring a
  separate radius, boundary, radial-order, and sampled-zero policy.

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
  Conv2d(3 -> 32, kernel=7, stride=1, padding=3, bias=False)
  Norm(32)
  ActivationPolicy(32)

ResBlock(in_channels, out_channels, downsample):
  main:
    Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
    ActivationPolicy(out_channels)
    if downsample: FixedBinomialLowpassDownsample2x(out_channels)
    Conv2d(out_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
  skip:
    identity if not downsample and in_channels == out_channels
    otherwise ResNet-D-style projection:
      if downsample: FixedBinomialLowpassDownsample2x(in_channels)
      Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
      Norm(out_channels)
  output:
    ActivationPolicy(out_channels)(main + skip)

Encoder stages:
  stage 256: two ResBlocks at 32 channels, first downsample=False
  stage 128: two ResBlocks at 48 channels, first downsample=True
  stage 64: two ResBlocks at 64 channels, first downsample=True
  stage 32: two ResBlocks at 96 channels, first downsample=True

VAE heads:
  mu_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2, bias=True)
  logvar_head: Conv2d(96 -> 16, kernel=5, stride=1, padding=2, bias=True)
```

Decoder mirrors the encoder:

```text
Latent projection:
  Conv2d(16 -> 96, kernel=5, stride=1, padding=2, bias=False)
  Norm(96)
  ActivationPolicy(96)

UpResBlock(in_channels, out_channels, upsample):
  main:
    optional bilinear upsample(scale_factor=2, align_corners=False)
    Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
    ActivationPolicy(out_channels)
    Conv2d(out_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
    Norm(out_channels)
  skip:
    identity if not upsample and in_channels == out_channels
    otherwise ResNet-D-style up projection:
      if upsample: bilinear upsample(scale_factor=2, align_corners=False)
      Conv2d(in_channels -> out_channels, kernel=5, stride=1, padding=2, bias=False)
      Norm(out_channels)
  output:
    ActivationPolicy(out_channels)(main + skip)

Decoder stages:
  stage 32: two UpResBlocks at 96 channels, first upsample=False
  stage 64: two UpResBlocks to 64 channels, first upsample=True
  stage 128: two UpResBlocks to 48 channels, first upsample=True
  stage 256: two UpResBlocks to 32 channels, first upsample=True

Output:
  Conv2d(32 -> 3, kernel=5, stride=1, padding=2, bias=True)
  zero-initialize weight and bias
```

Allowed first-run operations:

- odd square `Conv2d` with 7x7 only in the stem and 5x5 everywhere else;
- fixed fieldwise anti-aliased 2x downsampling for encoder stage transitions,
  used in both residual branches at their literature-consistent branch-local
  locations, followed by stride-1 odd square convolution;
- bilinear upsampling plus convolution for upsampling;
- ResNet-like residual adds with identity skips when shape/channel match;
- ResNet-D/anti-aliased-style projection skips: fixed spatial resampling first,
  then 5x5 spatial projection convolution, never 1x1 pointwise projections;
- spatial Gaussian VAE latent map;
- scalar gated activation policy defined below.

Output-head policy:

- do not apply a final `tanh`, sigmoid, or clamp in the model forward path;
- initialize the final RGB convolution weight and bias to zero, so the initial
  reconstruction is the normalized midpoint `0.0` and early training is stable;
- compute L1 against raw `x_hat` and `x_clean` in normalized `[-1, 1]`
  coordinates;
- compute SSIM, PSNR, image saving, and qualitative artifacts after projecting
  model output to image coordinates with
  `x_hat_img = clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)`;
- log output range telemetry, including `x_hat_min`, `x_hat_max`, and fraction
  of pixels below `-1` or above `1`, so the run exposes boundary behavior
  instead of hiding it behind a saturating output nonlinearity.

Residual policy:

- first-run residual connections are required and ResNet-like;
- no ReZero/Fixup/SkipInit learned residual scaling in spec 0001 unless a later
  spec explicitly adds it;
- projection skips are not naive one-shot channel adapters. They use explicit
  spatial resampling followed by odd 5x5 convolutions so they have a direct
  fixed-resampling-plus-repo-owned-SO2-convolution counterpart;
- encoder stage-transition blocks use branch-local fieldwise anti-aliased
  downsampling, following ResNet-D / BlurPool style rather than a pre-split
  downsample:
  the main branch replaces learned stride with
  `Conv5x5(stride=1) -> ActivationPolicy -> fixed_downsample_2x`, while the
  skip branch uses `fixed_downsample_2x -> Conv5x5` when spatial size changes;
- downsampling must not be hidden inside a learned stride or a one-off shortcut
  adapter;
- the fixed downsample operator is chosen from the future `SO(2)` side first:
  it must be a fieldwise spatial operator mapping a `FieldType` to the same
  `FieldType`, applying the same scalar spatial resampling to every fiber
  component and never mixing channels/frequencies;
- spec 0001 locks a repo-owned 5x5 separable binomial low-pass filter followed
  by decimation by 2:
  `kernel_1d = [1, 4, 6, 4, 1] / 16`, `kernel_2d = outer(kernel_1d, kernel_1d)`,
  zero padding `2`, and decimation by taking stride-2 samples. The Torch
  implementation may use fixed grouped `conv2d(..., groups=C, stride=2)` as an
  implementation detail, but this is a fixed fieldwise resampling operator, not
  a learned grouped/depthwise convolution;
- the fixed low-pass/downsample maps `(B, C, H, W)` to `(B, C, H/2, W/2)` for
  even `H,W`, preserves dtype/device where numerically safe, stores the filter
  as a non-trainable FP32 buffer, and applies the same scalar spatial operator
  independently to every future fiber component;
- resize/area-style scale-factor-0.5 fieldwise downsampling is moved to a later
  fallback/spike only if the locked binomial operator fails a future SO(2)
  stage-transition equivariance test. It is not a spec 0001 benchmark axis;
- FLOPs for the chosen fixed downsample are reported separately from learned
  convolutions;
- fieldwise downsampling is representation-compatible because it acts as a
  scalar spatial operator tensored with the identity on fiber components, but it
  is still a sampled-grid approximation. Future `SO(2)` stage transitions must
  include measured equivariance-error tests rather than assume perfect
  continuous-grid behavior;
- decoder up-projection skip uses bilinear upsampling before the 5x5 projection
  conv;
- parameter/FLOP counting must include all residual skip projections and fixed
  resampling operators;
- the future `SO(2)` model must mirror residual topology with matching
  `FieldType`s before addition.

Analytic Conv2d baseline count target:

The locked non-equivariant topology above has the following analytic count for a
single `256x256` RGB sample. Count MACs as multiply-accumulates; if reporting
FLOPs with the common multiply-plus-add convention, use `FLOPs = 2 * MACs`.

| Count target | Value | Notes |
| --- | ---: | --- |
| Learned convolution count | 43 | Includes skip projections, VAE heads, and RGB head |
| Normalization module count | 40 | `GroupNorm` modules with affine parameters |
| Gate module count | 34 | One learned scalar gate per hidden activation site |
| Fixed resampling op count | 12 | Six branch-local downsample ops plus six bilinear upsample ops |
| Learned convolution parameters | 3,949,539 | Includes zero-initialized RGB head bias |
| GroupNorm affine parameters | 4,800 | `weight,bias` for every norm channel |
| Learned gate parameters | 4,096 | Per-channel `a,b` for 2,048 activation-channel instances |
| Total learned parameters | 3,958,435 | Convs + norms + gates |
| Learned convolution MACs/sample | 36,471,046,144 | `36.471` GMAC/sample |
| Learned convolution FLOPs/sample | 72,942,092,288 | `72.942` GFLOP/sample with `2*MAC` convention |
| Fixed resampling MACs/sample | 85,032,960 | Conservative grouped-5x5 downsample plus 4-tap bilinear upsample |
| Fixed resampling FLOPs/sample | 170,065,920 | `0.170` GFLOP/sample with `2*MAC` convention |
| Total MACs/sample with fixed resampling | 36,556,079,104 | `36.556` GMAC/sample |

Section-level learned-convolution count:

| Section | Learned conv params | Learned conv MACs/sample |
| --- | ---: | ---: |
| Stem | 4,704 | 308,281,344 |
| Encoder residual body | 1,811,200 | 17,013,129,216 |
| VAE heads | 76,832 | 78,643,200 |
| Decoder and RGB head | 2,056,803 | 19,070,992,384 |

Activation-memory planning target:

- summing all learned-conv output tensors once gives `36,110,336` elements per
  sample;
- this rough activation-output sum is `137.75 MiB/sample` in FP32 and
  `68.88 MiB/sample` in FP16;
- largest individual hidden maps are `32x256x256 = 2,097,152` elements
  (`8 MiB` FP32), `48x128x128 = 786,432` elements (`3 MiB` FP32),
  `64x64x64 = 262,144` elements (`1 MiB` FP32), `96x32x32 = 98,304`
  elements (`0.375 MiB` FP32), and latent `16x32x32 = 16,384` elements
  (`0.0625 MiB` FP32);
- this is not a full autograd peak-memory estimate. The benchmark must still
  measure `max_vram_allocated_mb`, `max_vram_reserved_mb`, and headroom on
  Kaggle.

Implementation requirements:

- the model-count CLI/test must write `benchmark/model_count.json` and compare
  learned parameters, learned-conv MACs, and fixed-resampling MACs against the
  target above;
- if the fixed binomial downsample is implemented separably, report both the
  actual implementation MACs and the conservative dense grouped-5x5 equivalent;
- any topology change that moves a resampling op, adds/removes a norm, changes
  a kernel size, or changes gate placement must update this count section in
  the same patch.

Banned first-run operations:

- FSQ or any vector-quantized bottleneck;
- PixelShuffle or sub-pixel convolution;
- nearest-neighbor upsampling in the comparable path;
- 1x1 pointwise convolutions;
- learned depthwise/grouped/MBConv/squeeze-excite/channel-attention operations;
- `BatchNorm2d`, `LayerNorm`, channel dropout, or arbitrary normalization that
  cannot be mapped to the future SO(2) field schedule. Baseline `GroupNorm` is
  required and the future SO(2) counterpart is repo-owned field-aware norm;
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
These learned `a_i,b_i` gate parameters are intentionally added to the baseline
as well as the future `SO(2)` scalar/trivial fields. The purpose is to restore
some pointwise activation expressivity that the equivariant model loses when it
cannot use arbitrary componentwise nonlinearities, while keeping scalar-field
nonlinear expressivity matched between models.

Baseline scalar gate:

```text
gate_i = sigmoid(a_i * x_i + b_i)
out_i = gate_i * x_i
```

Rules for the baseline scalar gate:

- `a_i` and `b_i` are learned scalar parameters per channel;
- initialize to ordinary SiLU/Swish behavior where possible
  (`a=1`, `b=0`);
- do not add scalar activation parameters only to one model. The non-equivariant
  baseline and future `SO(2)` scalar/trivial fields use the same learned
  pointwise scalar gate family;
- do not tie or group baseline activation parameters by future field schedule in
  the first run. Any grouped activation tying requires a later explicit
  ablation/spec.

Future `SO(2)` counterpart:

- scalar/trivial fields use the same learned pointwise scalar gate family;
- learned additive bias is allowed only on scalar/trivial output fields;
- nontrivial 2D irrep copies use a radial gate over an invariant norm;
- learned additive vector bias is forbidden on nontrivial irrep/vector fields;
- nontrivial radial gates are implemented and tested as part of the activation
  policy, but they are not applied to fake vector pairs in the first scalar
  Conv2d baseline.

For each future 2-channel irrep copy `v = (u, w)`:

```text
r = sqrt(||v||**2 + eps) = sqrt(u**2 + w**2 + eps)
gate = sigmoid(a_i * r + b_i)
out = gate * v
```

Rules for future radial gates:

- the two components in a vector pair must share the same gate;
- initialize future vector/irrep copies to the same neutral gate convention as
  scalar fields (`a=1`, `b=0`) unless a later spec changes this;
- future vector/irrep copies may have scalar gate bias `b_i`, but must not have
  an additive learned 2D vector bias because that would break `SO(2)`
  equivariance;
- `eps` is required for stable gradients near zero vector norm. It must be large
  enough to avoid FP16 underflow/instability in AMP runs, configured explicitly,
  and tested in local/benchmark smoke. First candidate: `eps = 1e-4`;
- no learned activation amplitude `gamma` is used in spec 0001. Amplitude is
  handled by convolutions and normalization affine parameters, and `gamma` is
  reserved for a later ablation if the equivariant model is underpowered;
- gate parameters are included in trainable parameter counts and reported
  separately as a count and percentage of the model;
- implement this as an explicit `GatedScalarActivation`,
  `RadialGate`, and `ActivationPolicy` module using a central field schedule,
  not ad hoc reshaping inside model blocks;
- add a unit test that rotates synthetic vector pairs and verifies
  `activation(rho(theta) v) == rho(theta) activation(v)` within tolerance.

Normalization contract for the real run:

- the non-equivariant Conv2d baseline uses ordinary `torch.nn.GroupNorm` with
  affine parameters;
- default baseline groups: `num_groups = 8` for hidden widths
  32/48/64/96 and 16 latent-projection channels where normalization is applied;
- the future SO(2) model uses a repo-owned field-aware norm, not arbitrary raw
  GroupNorm over tensor channels;
- scalar/trivial fields may use additive affine bias;
- nontrivial frequency-1/frequency-2 vector fields may use invariant scalar
  scale parameters, but no additive learned vector bias;
- vector/irrep normalization uses invariant energy over whole irrep copies, for
  example RMS over `(copy, component, spatial)` groups chosen in the field
  schedule. It must never split a 2D irrep copy or group frequency-1 and
  frequency-2 components as if they were ordinary channels;
- normalization placement is after learned convolutions and before activation;
- VAE `mu_head`, `logvar_head`, and the final RGB output head do not use
  normalization;
- when a projection skip has a learned projection convolution at a location where
  the matching main branch is normalized before residual addition, normalize the
  projection branch before the add as well;
- convolution bias is disabled when immediately followed by normalization;
  scalar affine bias lives in the normalization or scalar activation. Learned
  biases remain allowed for scalar-only heads that are not followed by
  normalization.

## Objective Contract

Use a normal denoising VAE with a composite reconstruction objective:

```text
z = mu + exp(0.5 * logvar) * eps
l1_loss = mean(abs(x_hat - x_clean))
ssim_loss = 1 - ssim(project_for_ssim(x_hat), project_for_ssim(x_clean))
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

Precision and autograd policy:

- Mirror the useful FSQ precision structure as the conservative candidate:
  allow AMP/fp16 for the main model convolutional forward when the runtime
  benchmark selects AMP, but keep numerically sensitive islands in FP32.
- Do not assume the conservative split is fastest or necessary. The Kaggle
  runtime benchmark must compare safe precision placements and select the
  fastest one that passes numerical checks.
- Run the corruption module under `torch.no_grad()` and compute HED/OD color
  transforms, logarithms, exponentials, and random stain/noise draws in FP32.
  Corruption is data augmentation, not a differentiable model component.
- Run VAE posterior arithmetic in FP32 with gradients enabled:
  `logvar` clamp, `exp(0.5 * logvar)`, latent sampling, and KL computation.
- Run SSIM, L1, KL, beta weighting, and total loss composition in FP32 outside
  autocast. SSIM buffers/constants are FP32.
- Run radial-gate norm/sigmoid arithmetic in FP32 when AMP is enabled, using the
  configured `radial_gate_eps`, then return to the surrounding model dtype if
  needed. The gate remains differentiable with respect to the input field and
  gate parameters.
- Cast the model reconstruction output to FP32 before losses and metrics.
- Do not wrap training model forward, VAE sampling, losses, or SO(2) basis
  expansion in `torch.no_grad()`. Fixed basis buffers are non-trainable, but
  expansion coefficients require gradients.
- Use `torch.no_grad()` for metric accumulation, range/telemetry summaries,
  validation/evaluation passes, fixed-patch artifact generation, and checkpoint
  serialization helpers.
- Unlike the historical branchless FSQ validation path, `eval_clean` must not
  call the corruptor or consume corruption RNG. Deterministic corruption is used
  only in `eval_corrupted`.

Precision candidates for the Kaggle runtime benchmark:

- `amp_off_fp32`: full FP32 training step, used as the correctness and stability
  baseline.
- `amp_conservative`: main convolutional forward under AMP/fp16; corruption,
  posterior/KL, scalar/radial gate sigmoid arithmetic, SSIM, L1, and total loss
  in FP32.
- `amp_scalar_gate_relaxed`: same as `amp_conservative`, except the
  non-equivariant scalar gate sigmoid/multiply may run in the surrounding AMP
  dtype. This policy is eligible only for the scalar Conv2d baseline and only if
  paired numerical checks against `amp_off_fp32` pass. Posterior sampling,
  `logvar`, KL, SSIM/L1/loss, corruption, and future radial-gate norm/sigmoid
  arithmetic must remain FP32 in spec 0001.

Do not relax posterior/KL/loss/corruption or radial-gate norm/sigmoid numerics
in spec 0001. A broader precision ablation requires a later spec.

Corruption strategy candidates for the Kaggle runtime benchmark:

- `branchless_all`: compute corrupted images for the full batch, sample a mask,
  and select corrupted versus clean tensors with `torch.where`, matching the
  compile-friendly historical FSQ pattern.
- `indexed_masked`: sample a mask, corrupt only selected samples, and scatter
  them back into the batch. Accept this only if `torch.compile` stays stable and
  throughput improves.
- Both strategies must produce the same training distribution, support
  reproducible RNG, and preserve the validation rule that `eval_clean` consumes
  no corruption RNG.
- equivalence tests must key randomness by `sample_id`, rank, and optimizer
  step, then verify that `branchless_all` and `indexed_masked` produce the same
  Bernoulli corruption decisions, the same HED/noise parameters for corrupted
  samples, unchanged clean samples, no RNG consumption in `eval_clean`, and
  stable compile behavior across varying mask counts.

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

All values that affect the experiment must live in versioned JSON config files,
not hidden inside model or CLI code. JSON is the first-run config format because
it can be parsed and written with the Python standard library on offline Kaggle
kernels. `uv`, `pyproject.toml`, and `uv.lock` remain the Python environment and
dependency source of truth; they are orthogonal to experiment config files. CLI
flags may override config values only when the override is recorded in the run
config snapshot.

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

Optimizer parameter groups:

- learned convolution kernels, VAE head weights, final RGB head weights, and
  future `SO(2)` kernel expansion coefficients use the base learning rate and
  configured weight decay;
- additive biases, GroupNorm/field-norm affine parameters, and activation gate
  parameters use `weight_decay = 0.0`;
- activation gate parameters `a_i` and `b_i` use `lr_multiplier = 0.5` for the
  first run;
- this replaces the historical FSQ shape-only grouping with a semantic grouping:
  future `SO(2)` expansion coefficients may be stored as 1D tensors but still
  count as learned kernel weights and should receive weight decay unless a later
  spec overrides it;
- log gate parameter min/max and gate saturation summaries for scalar and radial
  gates.

Gate-health benchmark before the first full run:

- treat learned gate parameters as monitored capacity, not as an activation
  ablation in spec 0001;
- during the short real-data Kaggle debug/benchmark path, log gate behavior at
  fixed intervals for every gated activation module;
- write `metrics/gate_health.csv` with at least
  `run_name,optimizer_step,module,gate_kind,num_channels,num_elements,a_min,a_max,a_mean,a_std,b_min,b_max,b_mean,b_std,max_abs_a,max_abs_b,gate_mean,gate_std,gate_p01,gate_p50,gate_p99,frac_gate_lt_0_01,frac_gate_gt_0_99,worst_channel_frac_gate_lt_0_01,worst_channel_frac_gate_gt_0_99,dead_channel_count,input_rms,output_rms,output_input_rms_ratio,a_grad_norm,b_grad_norm,a_update_to_param_norm,b_update_to_param_norm,gate_health_status`;
- write `benchmark/gate_health_summary.json` with per-module worst-case
  saturation, non-finite counts, largest absolute `a`/`b`, dead-channel counts,
  zero-gradient counts, final input/output RMS ratio, and an overall
  `pass|warn|fail` status;
- compute `*_update_to_param_norm` as
  `update_norm / max(parameter_norm, 1e-8)` so zero-initialized or near-zero
  parameters have a defined denominator;
- gate-health warning thresholds: `max_abs_a > 10`, `max_abs_b > 10`,
  `max(frac_gate_lt_0_01, frac_gate_gt_0_99) >= 0.80`,
  `dead_channel_count > 0`, `output_input_rms_ratio < 1e-2`, or zero
  `a_grad_norm + b_grad_norm` for three consecutive logged intervals;
- gate-health failure thresholds: any non-finite gate value, parameter, input,
  output, gradient, or update; `max_abs_a > 20` or `max_abs_b > 20`;
  `max(frac_gate_lt_0_01, frac_gate_gt_0_99) >= 0.95` for three consecutive
  logged intervals; `dead_channel_count > max(1, 0.10 * num_channels)`;
  `output_input_rms_ratio < 1e-3` for three consecutive logged intervals in a
  hidden block; or any gate-health status explicitly marked `fail`;
- do not start the first full training run unless
  `benchmark/gate_health_summary.json` has overall status `pass`. A `warn`
  status requires inspection and a spec/config update before full training;
- do not use the gate-health benchmark to choose among many nonlinearities
  unless a later spec explicitly opens an activation ablation.

Runtime benchmark requirement before the first full Kaggle run:

- the benchmark is a short decision run, not training; it must stop after fixed
  warmup/measured steps and must not tune model quality;
- use the real train and validation data loaders and the real training step;
- benchmark two accelerator modes:
  `single_visible_t4` and `dual_t4_ddp`;
- `single_visible_t4` may run inside the dual-T4 Kaggle machine by setting
  visible devices to one GPU and `world_size = 1`;
- `dual_t4_ddp` must launch with two ranks, restore the historical
  `torchrun --standalone --nproc_per_node=2` behavior or an equivalent
  self-spawn implementation, and record `world_size = 2`;
- the Kaggle kernel metadata must request `machine_shape = "NvidiaTeslaT4"`
  before the remote benchmark is pushed. This value was verified on 2026-06-11
  by pulling metadata for the existing `maximusshtefan/non-eq-vae` notebook that
  the Kaggle UI showed as GPU T4 x2. Because the metadata value does not encode
  visible device count, `dual_t4_ddp` rows must still verify
  `cuda_device_count == 2`, two T4 names, `world_size == 2`, and
  `nproc_per_node == 2` at runtime;
- for each GPU configuration, benchmark `torch.compile` off/on where the runtime
  supports it;
- within the AMP/precision axis, compare the named precision policies
  `amp_off_fp32`, `amp_conservative`, and `amp_scalar_gate_relaxed`;
- compare the corruption execution strategies `branchless_all` and
  `indexed_masked`; keep the branchless path unless masked indexing is compile
  stable, preserves RNG semantics, and is measurably faster;
- for each row, record warm steady-state samples/sec, step time, compile
  overhead, max VRAM, largest stable per-device batch, global batch,
  `amp_step_skipped` count, gate-health warning count, and any compile/DDP
  failure;
- batch size is selected from VRAM and throughput evidence for each runtime
  configuration, not hard-coded from the historical FSQ run.

Valid runtime matrix rows:

| AMP enabled | Precision policy | Compile | Corruption strategy |
| --- | --- | --- | --- |
| false | `amp_off_fp32` | false/true | `branchless_all` / `indexed_masked` |
| true | `amp_conservative` | false/true | `branchless_all` / `indexed_masked` |
| true | `amp_scalar_gate_relaxed` | false/true | `branchless_all` / `indexed_masked` |

Invalid rows must not be emitted, for example `amp_off_fp32` with AMP enabled or
`amp_conservative` with AMP disabled.

Benchmark budget and reset rules:

- default candidate per-device batch sizes:
  `[4, 8, 12, 16, 24, 32, 48, 64]`;
- each row starts from identical model weights, optimizer state, scaler state,
  beta/LR scheduler state, data order, and RNG seeds;
- each row uses `warmup_steps = 3`, `measured_steps = 12`, and `repeats = 1`;
  after a row is selected as a top candidate, rerun it with
  `warmup_steps = 5`, `measured_steps = 25`, and `repeats = 1`;
- if `torch.compile` needs compilation, report compile/startup time separately
  from steady-state step time;
- OOM rows are valid failure rows: record the attempted per-device batch size,
  the exception class/message hash, max allocated/reserved memory if available,
  and continue with the next smaller candidate;
- selected rows must leave at least 10 percent VRAM headroom after warmup and
  measured steps;
- any row with non-finite loss, non-finite gradients, an AMP skipped step,
  DDP failure, compile failure, repeated graph breaks/recompiles after warmup,
  or gate-health status `fail` is ineligible;
- `indexed_masked` must improve measured steady-state samples/sec by at least
  5 percent over `branchless_all` in the same accelerator/precision/compile
  setting, or else `branchless_all` remains selected;
- a faster AMP/compile/precision row must pass paired numerical checks against
  `amp_off_fp32` eager on the same fixed batches before it is eligible.

Paired numerical checks:

- for every non-FP32 candidate, run three fixed-seed benchmark batches against
  `amp_off_fp32` eager with identical model initialization, data order,
  corruption decisions, and latent noise;
- log absolute and relative deltas for total loss, reconstruction loss, L1,
  SSIM loss, KL, gradient norm, output range, `mu` stats, `logvar` stats,
  `logvar_clamp_count`, gate-health summary, and parameter-update norm;
- default pass thresholds are: no non-finite values, no AMP skipped step,
  absolute loss/reconstruction/L1/SSIM-loss delta `<= 1e-3` or relative delta
  `<= 5e-3`, KL relative delta `<= 1e-2`, gradient-norm relative delta
  `<= 0.05`, and parameter-update-norm relative delta `<= 0.05`;
- if a threshold is too strict for a future verified reason, update this spec
  before selecting that runtime.

Dataloader benchmark requirement:

- before selecting a runtime, write `benchmark/dataloader_matrix.csv` for train
  and validation shards on real Kaggle data;
- record at least
  `run_name,accelerator_mode,world_size,rank,split,num_workers,prefetch_factor,pin_memory,batch_size,batches_measured,batch_fetch_ms_p50,batch_fetch_ms_p95,h2d_ms_p50,h2d_ms_p95,loader_samples_sec,trainer_samples_sec,data_wait_fraction_p50,data_wait_fraction_p95,rank_sample_count,dropped_sample_count,status`;
- a runtime is ineligible if validation loading is unmeasured, any rank fails,
  rank sample counts differ beyond one batch, `data_wait_fraction_p95 > 0.20`,
  or loader throughput is below `1.25 * trainer_samples_sec` for the selected
  training row.

The selected baseline runtime must be recorded in the resolved config. Use
`per_device_batch_size`, `global_batch_size`, `mixed_precision.enabled`, and
`torch_compile.enabled`, plus explicit `precision.policy` and
`corruption.strategy`; do not leave the batch-size, precision, or corruption
execution meaning ambiguous.

`benchmark/model_count.json` required shape:

```json
{
  "status": "pass",
  "config": "invoked config path",
  "input_shape": [1, 3, 256, 256],
  "learned_convolution_count": 43,
  "normalization_module_count": 40,
  "gate_module_count": 34,
  "fixed_resampling_op_count": 12,
  "learned_convolution_parameters": 3949539,
  "groupnorm_affine_parameters": 4800,
  "learned_gate_parameters": 4096,
  "total_learned_parameters": 3958435,
  "learned_convolution_macs_per_sample": 36471046144,
  "fixed_resampling_macs_per_sample": 85032960,
  "total_macs_per_sample_with_fixed_resampling": 36556079104,
  "activation_output_elements_per_sample": 36110336,
  "matches_spec_target": true
}
```

`benchmark/runtime_matrix.csv` required columns:

```text
run_name,row_id,accelerator_mode,machine_shape,visible_device_count,cuda_device_count,gpu_names,ddp_backend,world_size,nproc_per_node,precision_policy,amp_enabled,torch_compile_enabled,corruption_strategy,per_device_batch_size,global_batch_size,gradient_accumulation_steps,warmup_steps,measured_steps,repeats,compile_startup_sec,steady_step_ms_p50,steady_step_ms_p95,samples_sec,trainer_samples_sec,max_vram_allocated_mb,max_vram_reserved_mb,vram_headroom_fraction,amp_step_skipped_count,gate_health_status,gate_health_warning_count,numerical_check_status,data_wait_fraction_p95,oom,status,failure_kind,failure_message_hash
```

`benchmark/selected_runtime.json` required shape:

```json
{
  "status": "pass",
  "selected_row_id": "string",
  "accelerator_mode": "single_visible_t4 or dual_t4_ddp",
  "machine_shape": "NvidiaTeslaT4",
  "world_size": 2,
  "nproc_per_node": 2,
  "gpu_names": ["..."],
  "per_device_batch_size": 0,
  "global_batch_size": 0,
  "gradient_accumulation_steps": 1,
  "optimizer_updates_per_epoch": 0,
  "lr_warmup_steps": 0,
  "beta_warmup_steps": 0,
  "mixed_precision": {"enabled": false, "policy": "amp_off_fp32"},
  "torch_compile": {"enabled": false, "backend": "eager-or-inductor"},
  "corruption": {"strategy": "branchless_all"},
  "throughput": {
    "samples_sec": 0.0,
    "steady_step_ms_p50": 0.0,
    "compile_startup_sec": 0.0,
    "estimated_10_epoch_wall_time_sec": 0.0
  },
  "safety": {
    "numerical_check_status": "pass",
    "gate_health_status": "pass",
    "dataloader_status": "pass",
    "amp_step_skipped_count": 0
  }
}
```

For `selected_runtime.json`, `world_size` and `nproc_per_node` are numeric:
`1` for `single_visible_t4`, `2` for `dual_t4_ddp`.

Required first-run budget defaults:

| Config | Batch size | Train steps | Validation interval | Checkpoint interval |
| --- | ---: | ---: | ---: | ---: |
| `non_eq_vae_debug_cpu.json` | 2 global | 8 | 4 | 4 |
| `non_eq_vae_kaggle_runtime_benchmark.json` | searched per device | short fixed benchmark steps | optional one fixed validation micro-pass | none except benchmark summary |
| `non_eq_vae_kaggle_debug.json` | benchmarked per device | 200 | 50 | half epoch or 100 steps |
| `non_eq_vae_kaggle_tiny_overfit.json` | selected runtime | 300 on 32 fixed real patches | 50 | 100 |
| `non_eq_vae_baseline.json` | benchmark-selected per device | 10 epochs | half epoch | half epoch |

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

- `config_resolved.json`: full config after CLI overrides;
- `metrics/train_steps.csv`: one row per logged train step with at least
  `run_name,event_id,batch_attempt,optimizer_step,split,loss,recon_loss,l1_loss,ssim_loss,ssim_metric,mae,mse,psnr,kl_loss,beta,lr,grad_norm,batch_size,precision_policy,amp_enabled,torch_compile_enabled,corruption_strategy,amp_step_skipped,mu_mean,mu_std,mu_min,mu_max,logvar_mean,logvar_std,logvar_min,logvar_max,logvar_clamp_count,x_hat_min,x_hat_max,frac_x_hat_lt_minus1,frac_x_hat_gt_1`;
- skipped AMP rows are logged as batch-attempt events with
  `amp_step_skipped = 1`; they do not increment `optimizer_step` and do not
  trigger optimizer-step-based schedules, validation, or checkpointing;
- `metrics/validation_steps.csv`: one row per validation event with at least
  `run_name,optimizer_step,split,view,n,mse_mean,mae_mean,psnr_mean,ssim_mean,kl_mean,mu_mean,mu_std,logvar_mean,logvar_std,logvar_clamp_count,x_hat_min,x_hat_max,frac_x_hat_lt_minus1,frac_x_hat_gt_1`;
- `eval/per_image_metrics.csv`: one row per evaluated patch with at least
  `sample_id,split,view,wsi_id,label,x,y,mse,mae,psnr,ssim`;
- `eval/summary.json`: mean, standard deviation, and `n` for every metric,
  grouped by `split` and `view`;
- `artifacts/manifest.json`: paths and provenance for every generated figure;
- `benchmark/model_count.json`: analytic/implementation model count comparison
  including learned parameters, learned-conv MACs, fixed-resampling MACs,
  normalization/gate parameters, and pass/fail status against this spec;
- `benchmark/runtime_matrix.csv`: one row per benchmarked runtime configuration;
- `benchmark/selected_runtime.json`: selected accelerator, compile, AMP, and
  batch-size decision for the first full run, including selected
  `precision.policy` and `corruption.strategy`;
- `benchmark/dataloader_matrix.csv`: real train/validation loader, transfer,
  throughput, wait-fraction, and rank-balance measurements;
- `benchmark/numerical_checks.csv`: paired fixed-batch deltas against
  `amp_off_fp32` eager for precision/compile candidates;
- `metrics/gate_health.csv`: per-module gate parameter, saturation, RMS, and
  gradient/update telemetry from debug and benchmark runs;
- `benchmark/gate_health_summary.json`: gate-health pass/warn/fail summary used
  before the first full run;
- `benchmark/tiny_overfit_summary.json`: selected-runtime real-patch overfit
  sanity summary before the first full run;
- `checkpoints/step_*.pt`: model, optimizer, scheduler, beta scheduler, scaler
  if present, current step, config hash, and RNG state.

`logvar_clamp_count` must be logged whenever any values are clamped.

## Fixed 25-Patch Protocol

The qualitative 25-patch set must be deterministic and shared by the baseline
and future `SO(2)` model.

Selection policy for `configs/spec0001/fixed_25_validation_patches.json`:

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
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
  --data-root auto \
  --output configs/spec0001/fixed_25_validation_patches.json
```

This generator requires access to the real validation CSV and is therefore a
data-access step, not a pure offline local test. Local synthetic tests may use a
separate generated synthetic selector under `runs/` but must never overwrite the
canonical fixed-25 config.

## Fixed 32-Train Tiny-Overfit Protocol

The tiny-overfit sanity check must not reuse the fixed 25-patch validation
artifact set. It uses a separate deterministic train-patch selector so the
validation qualitative set remains a held-out visual protocol.

Selection policy for `configs/spec0001/fixed_32_train_overfit_patches.json`:

1. Use the train CSV from `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
2. Exclude any WSI listed in `docs/data/ubc_ocean_masked_holdout_ids.csv`.
3. Compute `sha256("20260611:tiny-overfit:{wsi_id}:{label}:{x}:{y}")`.
4. Sort by digest, then by `wsi_id,label,x,y`.
5. Select the first 32 rows.
6. Store the ordered 32 selectors with `wsi_id,label,x,y,source_split`.

Tiny-overfit commands must fail if the fixed 32-train config is missing, has any
validation row, or contains a count other than 32.

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
- `src/eqvae/cli/`: `smoke`, `model_count`, `train`, `benchmark_runtime`,
  `select_fixed_patches`, `evaluate`, and `artifacts` entry points.

Required config files:

- `configs/spec0001/non_eq_vae_baseline.json`;
- `configs/spec0001/non_eq_vae_debug_cpu.json`;
- `configs/spec0001/non_eq_vae_kaggle_debug.json`;
- `configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`;
- `configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json`;
- `configs/spec0001/ubc_ocean_masked_holdout_test.json`;
- `configs/spec0001/fixed_32_train_overfit_patches.json`;
- `configs/spec0001/fixed_25_validation_patches.json`.

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

- configs are JSON by contract for spec 0001 and must use only the Python
  standard-library `json` parser/writer;
- do not add a YAML parser solely for experiment config files in spec 0001;
- repo-owned Torch SSIM must be implemented under `src/eqvae`; do not import
  `pytorch-msssim` in spec 0001 code unless a later spec deliberately changes
  the offline/compiled SSIM policy;
- if `pytorch-msssim` remains in `pyproject.toml` as historical dependency debt,
  resolve it through the benchmark-unblock route in
  `docs/specs/0002-strict-python-quality-gate.md`: extract/quarantine
  historical `src/nn`, remove the direct dependency, and refresh `uv.lock` in
  one cleanup patch before benchmark CLIs are implementation-ready.

Local CPU smoke policy:

- CPU smoke tests are shape/contract tests, not speed benchmarks;
- CPU `torch.compile` tests may use tiny synthetic batches and must have bounded
  step counts so they do not turn into long local training jobs;
- CPU float16 smoke is allowed to be a narrow dtype-path check with documented
  tolerances or explicit skips for unsupported CPU operations;
- GPU speed, AMP, and DDP behavior are decided only by the permission-gated
  Kaggle runtime benchmark.

Implementation milestones before broad coding:

1. Spec relock slice: implementation count verification, future SO(2) count
   ceiling, ResNet-like residual confirmation, package/import policy, JSON
   config policy, quality-debt route, fixed validation/tiny-overfit selector
   plan, artifact protocol, and final clean-context spec review.
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
7. Artifact/evaluation slice: fixed validation/tiny-overfit selectors,
   evaluator, boxplots, dashboards, rotated/latent artifacts.
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
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --batch-size 2 \
  --compile inductor \
  --dtype float32
```

Local CPU float16 smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.smoke \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --batch-size 1 \
  --max-steps 1 \
  --dtype float16
```

Debug train from scratch:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
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
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
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
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
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

Local synthetic benchmark schema smoke:

```bash
PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.model_count \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --output runs/local/spec0001-runtime-benchmark/benchmark/model_count.json

PYTHONPATH=src uv run --locked --no-sync python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_debug_cpu.json \
  --data synthetic \
  --device cpu \
  --output-dir runs/local/spec0001-runtime-benchmark \
  --run-name spec0001_cpu_runtime_benchmark \
  --max-benchmark-rows 2 \
  --warmup-steps 1 \
  --measured-steps 2
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
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
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
python -m eqvae.cli.model_count \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --output /kaggle/working/runtime_benchmark/benchmark/model_count.json

python -m eqvae.cli.benchmark_runtime \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working/runtime_benchmark \
  --run-name non_eq_vae_spec0001_runtime_benchmark
```

Kaggle selected-runtime debug command that must run after
`benchmark/selected_runtime.json` is written:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_debug.json \
  --runtime-config /kaggle/working/runtime_benchmark/benchmark/selected_runtime.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --output-dir /kaggle/working/selected_runtime_debug \
  --run-name non_eq_vae_spec0001_selected_runtime_debug \
  --max-train-steps 200 \
  --max-val-steps 20 \
  --save-every-steps 100
```

Kaggle tiny-overfit command that must pass before the first 10-epoch run:

```bash
python -m eqvae.cli.train \
  --config configs/spec0001/non_eq_vae_kaggle_tiny_overfit.json \
  --runtime-config /kaggle/working/runtime_benchmark/benchmark/selected_runtime.json \
  --data ubc-pre-shuffled \
  --data-root auto \
  --fixed-train-patches configs/spec0001/fixed_32_train_overfit_patches.json \
  --output-dir /kaggle/working/tiny_overfit \
  --run-name non_eq_vae_spec0001_tiny_overfit \
  --max-train-steps 300 \
  --max-val-steps 20 \
  --save-every-steps 100
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
   convs, learned grouped/depthwise convs, attention blocks, `BatchNorm2d`,
   `LayerNorm`, and arbitrary representation-breaking normalization, while
   requiring baseline `GroupNorm` in hidden/projection blocks;
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
10. model tests verify that the final RGB head is zero-initialized, that the
    initial reconstruction is all zeros within tolerance, and that the model
    forward path contains no final `tanh`, sigmoid, or clamp;
11. debug training completes from scratch and writes metrics, config, checkpoint,
   and RNG state;
12. resume training restores checkpoint, optimizer, scheduler/beta state, and RNG
   state;
13. AMP skipped-step behavior is tested or exercised so skipped steps do not
    advance optimizer-step counters, LR/beta schedules, validation, or
    checkpoint cadence;
14. the runtime benchmark CLI exists, runs on a tiny local synthetic budget, and
    writes `benchmark/runtime_matrix.csv`, `benchmark/selected_runtime.json`,
    `benchmark/model_count.json`, `benchmark/dataloader_matrix.csv`,
    `benchmark/numerical_checks.csv`, `metrics/gate_health.csv`, and
    `benchmark/gate_health_summary.json` with the expected schemas without
    requiring GPU or network access;
15. checkpoint retention keeps `best_model.pt`, the final checkpoint, and the
    latest four interval checkpoints;
16. evaluator writes per-image SSIM, MAE, MSE, PSNR and summary mean/std/`n`
    separately for `eval_clean` and fixed-seed `eval_corrupted`;
17. artifact writer emits metric boxplots, dashboard, fixed 25-patch
    reconstructions, rotated-input grids, rotated-input versus latent grids, and
    latent visualization placeholders or outputs;
18. offline selector tests use synthetic fixtures; the real
    `fixed_25_validation_patches.json` and
    `fixed_32_train_overfit_patches.json` generation are data-access steps that
    must be run on the real validation/train CSVs before Kaggle debug/full runs;
19. the fixed 25-patch config contains exactly 5 validation patches per label and
    all future qualitative commands read it rather than resampling; the fixed
    32-train tiny-overfit config contains exactly 32 train patches and no
    validation rows;
20. Kaggle debug kernel runs bundled repo code through the CLI, not notebook
    source or a GitHub-linked notebook;
21. `scripts/kaggle_kernel.sh push` rejects wrong dataset slugs, historical FSQ
    output sources, internet-enabled metadata, missing payloads, placeholder
    launchers, missing or wrong benchmark `machine_shape`, and missing
    single-visible versus dual-DDP launch-mode validation hooks;
22. runs without the sealed masked-WSI test shard are labeled
    train/validation-only and excluded from final paper claims;
23. `CURRENT.md`, `docs/specs/README.md`, and relevant workflow docs are updated
    with implementation status and verification results.

## Full Kaggle Run Acceptance Criteria

The first 10-epoch Kaggle baseline is not ready until:

1. local implementation acceptance passes;
2. the read-only Kaggle API preflight
   `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes its
   required auth/list/status/logs/dataset checks. If the quota endpoint warns,
   verify GPU quota in the Kaggle web UI before approving the remote benchmark
   push and record the warning in the run notes;
3. the user explicitly approves the remote Kaggle write/run;
4. the model-count command writes `benchmark/model_count.json` with status
   `pass` before runtime selection;
5. the short Kaggle runtime benchmark writes `benchmark/runtime_matrix.csv` and
   `benchmark/selected_runtime.json`, including AMP off/on, compile off/on,
   named precision-policy, and branchless-versus-indexed corruption evidence
   for single and dual T4;
6. the benchmark writes `benchmark/dataloader_matrix.csv` and
   `benchmark/numerical_checks.csv`; the selected row must have dataloader,
   numerical-check, and gate-health status `pass`;
7. the selected single/dual T4, per-device/global batch, AMP, compile,
   precision policy, and corruption strategy are copied into the resolved
   full-run config;
8. the gate-health benchmark writes `metrics/gate_health.csv` and
   `benchmark/gate_health_summary.json` without non-finite gate values,
   persistent near-total saturation, or unexplained near-zero hidden-block
   output/input RMS;
9. a selected-runtime real-data debug run completes 200 train steps, runs both
   `eval_clean` and fixed-seed `eval_corrupted`, writes at least one checkpoint,
   resumes once from that checkpoint, and emits nonblank fixed-patch artifacts;
10. a selected-runtime tiny-overfit run on 32 fixed real patches writes
   `benchmark/tiny_overfit_summary.json` with finite losses, final smoothed L1
   and reconstruction loss at least 5 percent below their initial smoothed
   values, PSNR or SSIM improved over the zero-head baseline, no pathological
   `logvar_clamp_count`, and gate-health status not worse than `warn`;
11. the baseline run uses the selected runtime config, validates/checkpoints every
   half epoch, and keeps the declared checkpoint retention.

## Open Questions And Gates

Implementation-relock blockers:

1. Final channel/future-field schedule: the Conv2d baseline count is now
   recorded in this spec. Before the `SO(2)` implementation is locked, the
   steerable basis/count tool must show the future field multiplicities can stay
   at or below the Conv2d baseline's learned parameter count without exceeding
   the Kaggle memory budget.
2. Implementation model-count artifact: the benchmark implementation must write
   `benchmark/model_count.json` and verify the exact locked residual topology,
   projection shortcuts, GroupNorm/activation gates, fixed binomial resampling
   FLOPs, and zero-initialized RGB head against the analytic target above.
3. Kaggle metadata enforcement: the workflow now records
   `machine_shape = "NvidiaTeslaT4"` for the T4 benchmark kernel. The
   implementation must enforce that metadata value before remote push and fail
   `dual_t4_ddp` benchmark rows unless runtime CUDA/DDP telemetry proves two T4
   devices and two ranks.
4. Final clean-context adversarial spec review must pass after the edits and
   implementation count, metadata, import, and quality routes are integrated.
5. Strict quality-debt route must follow spec 0002's benchmark-unblock route:
   extract any needed behavior into `src/eqvae`, remove/quarantine historical
   `main.py` / exploratory `src/nn` without leaving importable `.py` debt,
   remove the historical `pytorch-msssim` dependency, refresh `uv.lock`, and
   keep global Ruff/BasedPyright strictness intact.
6. JSON config/dependency policy must remain locked, or a later spec must
   explicitly justify changing config format and dependencies.
7. Package/import policy must be locked enough that the verification commands
   import `eqvae` without dependency sync.
8. Fixed-25 selector generation and baseline rotated/latent artifact semantics
   must remain exactly as specified above, or be revised before implementation.

Full-run blockers after implementation:

1. Runtime target: after the Kaggle benchmark matrix, should the full run use
   single GPU or dual T4 DDP, should AMP and/or `torch.compile` be enabled, and
   what are the selected precision policy, corruption execution strategy, and
   per-device/global batch size?
2. The selected runtime must be written to `benchmark/selected_runtime.json` and
   the resolved baseline config before the first 10-epoch Kaggle run.
3. Gate-health target: the short benchmark/debug path must show that learned
   gate `a,b` parameters do not create non-finite values, persistent saturation,
   or hidden-block collapse before the first full run.
4. Data/quality target: dataloader throughput, paired numerical checks,
   selected-runtime debug, checkpoint/resume, and tiny-overfit summaries must all
   pass before the first 10-epoch Kaggle run.

## Known Risks

- The future radial gate can suppress vector/irrep copies if initialized poorly.
  Initialize it with the accepted Swish-like `a=1,b=0` convention and test
  gradient flow before the `SO(2)` model depends on it.
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
