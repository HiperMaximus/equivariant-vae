# Equivariant VAE Transition Plan

Status: draft working plan
Last updated: 2026-06-11

## Purpose

Read this plan together with:

- `GOAL.md` for the repository north star.
- `CURRENT.md` for the current handoff state and next concrete steps.
- `docs/repo_goal_and_requirements.md` for issue-derived deliverables and
  acceptance gates.
- `docs/issue_image_inventory.md` for requirements derived from inspected
  GitHub issue screenshots.
- `docs/kaggle_cli_workflow.md` for CLI-managed Kaggle script execution.
- `docs/behavior_inventory_kaggle.md` for historical Kaggle data, training,
  resume, metric, and artifact behavior.
- `docs/decisions/README.md` for settled project decisions.
- `docs/agentic_review_workflow.md` for adversarial clean-context review before
  major architecture or workflow changes.
- `docs/spec_driven_development.md` and `docs/specs/` for spec-first
  implementation contracts.

The current Kaggle runs proved that the UBC-OCEAN patch pipeline, DDP loop,
stain/noise corruption, checkpointing, and evaluation artifact flow can work.
The latest model, however, is an FSQ autoencoder, not a normal VAE. That makes it
the wrong long-term baseline for an `escnn`/steerable comparison because too many
operations and assumptions do not translate cleanly.

This plan replaces the current non-equivariant experiment with a continuous
denoising VAE baseline whose operations are equivariant-translatable by
construction. The baseline should be the "shadow" of the future equivariant model:
the same data contract, layer schedule, losses, logging, and evaluation gates,
with only the layer factory and field representations swapped.

The target comparison is:

1. Non-equivariant denoising VAE baseline.
2. Steerable/equivariant denoising VAE with matched training, data, evaluation,
   and capacity reporting.

## References

- `escnn` repository: https://github.com/QUVA-Lab/escnn
- `escnn` docs: https://quva-lab.github.io/escnn/
- G-spaces and planar groups: https://quva-lab.github.io/escnn/api/escnn.gspaces.html
- `escnn.nn` modules, field types, nonlinearities, normalization:
  https://quva-lab.github.io/escnn/api/escnn.nn.html
- `R2Conv` implementation details, kernel size, rings, and frequency cutoff:
  https://quva-lab.github.io/escnn/_modules/escnn/nn/modules/conv/r2convolution.html
- `R2Upsampling` implementation and bilinear/nearest warning:
  https://quva-lab.github.io/escnn/_modules/escnn/nn/modules/rdupsampling.html

## Current Repo Facts To Resolve

- `README.md` has been reset to point at `GOAL.md`, this transition plan, the
  requirements tracker, and the SIPAIM paper workflow.
- The latest Kaggle notebook trains on 256x256 UBC-OCEAN patches.
- The latest Kaggle model is a deterministic FSQ autoencoder. It has no KL term
  and should not be reported as a VAE baseline.
- `main.py` is empty, and the last runnable notebook artifacts/evidence are the
  Kaggle notebook JSON files in `kaggle/train_runs`. Dataset-generation evidence
  lives in `kaggle/dataset_generation` and
  `kaggle/generate_dataset_Classification_With_Masks`. These need to become
  references, not executable source-of-truth files.
- `src/nn/layers.py` and `src/nn/resnet18.py` contain useful architectural
  explorations, but also operations that should not be carried forward blindly:
  GroupNorm, depthwise/MBConv, squeeze-excite, nearest upsampling, 1x1-heavy
  projections, and non-field-aware channel logic.
- Environment metadata now has a strict local source of truth:
  `pyproject.toml` for direct dependencies/tooling and `uv.lock` for the
  resolved CPU-only laptop environment. Kaggle-only dependencies from notebooks
  are not fully captured yet and should be added through a dedicated Kaggle
  group/bootstrap spec, not a root `requirements.txt`.

## Experiment Contract

Lock these before implementation:

1. Input size.
2. Latent shape.
3. Primary symmetry group.
4. Representation schedule.
5. Normalization policy.
6. Data split policy.
7. Fairness budget.

Current proposal after the 2026-06-11 spec correction:

- Continue the current UBC-OCEAN patch contract: 256x256 RGB patches normalized
  to `[-1, 1]`.
- Use an FSQ-successor spatial Gaussian latent target `(B, 16, 32, 32)` for the
  first comparable VAE, preserving spatial coherence for the future `SO(2)`
  comparison while removing the FSQ quantizer and its learned scalar `s`
  bottleneck trick.
- First implementation target: continuous `SO(2)` steerability via
  `rot2dOnR2(N=-1, maximum_frequency=2)`.
- Use frequencies up to `L <= 2` in the steerable model.
- Treat compact `4x4` latents as later ablations unless explicitly relocked.
- Split by WSI/patient/site where metadata allows. Never split by patch.

## Non-Negotiable Design Rules

1. No FSQ, vector quantization, codebook histograms, `round`, discrete indices,
   or tanh-bounded bottleneck in the replacement baseline.
2. Use a normal VAE bottleneck:
   `z = mu + exp(0.5 * logvar) * eps`.
3. No PixelShuffle or sub-pixel convolution. Use bilinear upsampling followed by
   a convolution.
4. No nearest-neighbor upsampling in the equivariant path.
5. No raw `torch.nn.GroupNorm`, `BatchNorm2d`, `LayerNorm`, per-channel affine
   normalization, or channel dropout in the equivariant path.
6. No depthwise, grouped, MBConv, squeeze-excite, or channel-attention operations
   unless a steerable equivalent is identified and tested.
7. No arbitrary `view`, `rearrange`, `.chunk()`, channel slicing, or flattening on
   a `GeometricTensor` unless it is field-aligned and documented.
8. No layer enters the baseline until its equivariant replacement is known.
9. Kaggle notebooks must become launchers, not the source of truth.

Pointwise convolution policy:

- `R2Conv(kernel_size=1)` can be equivariant, so 1x1 is not mathematically
  impossible.
- For this experiment, 1x1 convolutions are banned initially because the user
  wants the non-equivariant baseline to avoid pointwise convs and because larger
  spatial kernels make the `L <= 2` steerable basis meaningful.
- Any future 1x1 exception must be listed in the config and mirrored exactly in
  the equivariant version.

## Target Symmetry Scope

The research target is continuous `SO(2)`.
The experiment asks whether a non-equivariant VAE and a comparable `SO(2)`
steerable VAE differ in usefulness for histopathology.

### First Target: Continuous SO(2) With L <= 2

- Use `rot2dOnR2(N=-1, maximum_frequency=2)`.
- RGB input is three trivial scalar fields.
- Hidden layers may contain frequency-0, frequency-1, and frequency-2 fields.
- The steerable model must log every field type, multiplicity, kernel size, and
  basis frequency cutoff.
- The non-equivariant baseline must use the same macro-topology, input size,
  downsampling depths, latent shape, kernel sizes, losses, training budget, and
  evaluation protocol.

### Optional Later Target: O(2)

- Use `flipRot2dOnR2(N=-1, maximum_frequency=2)` only after the `SO(2)` pipeline
  is working.
- Reflections may or may not be label-preserving in histopathology, so `O(2)` is
  an explicit ablation rather than the first target.

Continuous image-space gate:

- Continuous rotations require interpolation, padding/cropping, boundary masks,
  and tolerances. The evaluation protocol must measure transform-induced error on
  raw inputs before interpreting model equivariance error.

## Operation Translation Table

| Role | Current FSQ script | Replacement baseline | Equivariant target | Notes |
| --- | --- | --- | --- | --- |
| Encoder convolution | ResNet-style Conv2d blocks | Odd square Conv2d, usually 5x5 or 7x7 | `escnn.nn.R2Conv` | Use the same abstract layer schedule in both models. |
| Channel changes | 1x1 projections and pointwise-style mixing | Spatial kernels only at first | `R2Conv` between compatible `FieldType`s | 1x1 is possible but banned initially. |
| Residual adds | Standard tensor addition | Allowed when shapes match | GeometricTensor add with same `FieldType` | Projection branch must produce the same field type as the main branch. |
| Downsampling | Strided conv/avg-pool shortcuts | Strided odd-kernel conv, optionally anti-aliased | `R2Conv(stride=2)` plus equivariance tests | Downsampling can alias and damage equivariance. |
| Upsampling | PixelShuffle or nearest-like decoder code in older files | Bilinear scale factor + Conv2d | `R2Upsampling(mode="bilinear")` + `R2Conv` | Use uniform `scale_factor`, not arbitrary nonuniform `size`. |
| Bottleneck | FSQ discrete 16-level latents | Gaussian VAE latent map | SO(2) scalar plus irrep-aware latent statistics | Test latent statistics under transforms. |
| Activation | ReLU/SiLU everywhere | Gated scalar activation on all ordinary tensor channels | Same scalar gate family plus radial gates over vector norms for nontrivial fields | Treat SiLU as `x * sigmoid(x)`; do not add equivariant-only activation parameters or vector biases. |
| Normalization | GroupNorm in many blocks | Prefer no norm first | `IIDBatchNorm2d`, `GNormBatchNorm`, `NormBatchNorm`, field norm, or no norm | Raw GroupNorm is unsafe until proven. |
| Corruption | HED stain/noise corruption | Same corruption policy | Same corruption policy | For equivariance losses, use controlled/shared randomness. |
| Loss | Charbonnier + SSIM, no KL | `L1 + 0.1 * (1 - SSIM) + beta * KL` | Same scalar losses plus optional equivariance regularizer | Composite beta-VAE-style objective, not a strict Gaussian ELBO. |
| Equivariance metric | 25-patch artifact metric | Dataset-level metric plus qualitative artifacts | Dataset-level metric plus layer/full-model checks | 25 patches are qualitative only. |

## Baseline Architecture Direction

The baseline should be generated from an abstract layer schedule that the
equivariant model can reuse.

Recommended SO(2)-compatible schedule:

```text
Input: 256x256 RGB, normalized to [-1, 1]
Latent: 16 x 32 x 32

For 256x256:
  256 -> 128 -> 64 -> 32
```

Use the same spatial schedule for both models. The equivariant model defines
channel count through field multiplicities, while the baseline mirrors the
resulting tensor widths without imposing steerable kernel constraints.

First SO(2) field schedule:

```text
Stage 0: scalar fields only for RGB input.
Stage 1: scalar + frequency-1 vector fields.
Stage 2: scalar + frequency-1 + frequency-2 vector fields.
Stage 3+: same frequency set, larger multiplicities.
Latent: start with scalar/trivial spatial latents for the first complete run.
        Then add frequency-1/frequency-2 latent fields with invariant variance.
```

The non-equivariant baseline treats all channels as ordinary scalar tensor
channels. Its Conv2d layers may freely mix all channels, and its gated scalar
activation is applied componentwise. The scalar/F1/F2 schedule remains capacity
bookkeeping for the future `SO(2)` field multiplicities, not a restriction on
baseline channel mixing.

Recommended block pattern:

```text
ConvBlock:
  Conv2d(in -> out, kernel=5 or 7, stride=1 or 2, padding=same)
  activation policy
  Conv2d(out -> out, kernel=5 or 7, stride=1, padding=same)
  no first-run residual/ReZero/Fixup branch

VAE heads:
  mu_head: Conv2d(C -> latent_channels, kernel=5, padding=2)
  logvar_head: Conv2d(C -> latent_channels, kernel=5, padding=2)

Decoder:
  Conv2d(latent_channels -> C, kernel=5, padding=2)
  repeated: bilinear upsample x2 + ConvBlock
  final Conv2d(C -> 3, kernel=5, padding=2)
  tanh output
```

Initial latent should stay spatial, not flattened. A spatial latent map
translates more naturally to equivariant feature fields than a dense vector.

KL convention must be written in the config:

- First-run convention: average KL over batch, latent channels, and latent spatial
  positions.
- Do not switch conventions between baseline and equivariant runs.

## VAE Objective Contract

The first replacement model is a normal denoising VAE, not an ad hoc stochastic
autoencoder.

Training input/target:

- `x_clean` is the clean patch normalized to `[-1, 1]`.
- `x_in = corrupt(x_clean)` is passed to the encoder during training.
- The decoder target is always `x_clean`.

First-run loss:

```text
l1_loss = mean(abs(x_hat - x_clean))
ssim_loss = 1 - ssim(x_hat, x_clean)
recon_loss = l1_loss + 0.1 * ssim_loss
kl_element = -0.5 * (1 + logvar - mu ** 2 - exp(logvar))
kl_loss = mean(kl_element)
beta = configured_linear_warmup_to_1
loss = recon_loss + beta * kl_loss
```

Interpretation:

- `recon_loss` is a composite image-fidelity objective chosen to remain close to
  the successful FSQ training signal while adding a proper Gaussian latent and
  KL term.
- `kl_loss` is measured per latent scalar. This keeps beta comparable if latent
  spatial size or channel count changes in an ablation.
- SSIM is part of the first-run objective with weight `0.1`; PSNR, MAE, MSE, and
  SSIM are still logged as metrics.
- Beta warms up over the first full epoch for epoch-based runs and over the
  first 10 percent of optimizer steps only for tiny step-limited debug runs.
- Do not call this a strict ELBO without qualification.

Validation metrics:

- Report clean autoencoding: encoder input `x_clean`, target `x_clean`.
- Report deterministic denoising: encoder input `corrupt(x_clean)` with a fixed
  validation seed/config, target `x_clean`.
- Report reconstruction metrics and KL terms for both views.

## Equivariant Architecture Direction

Core building blocks:

- Inputs are trivial RGB fields:
  `FieldType(gspace, 3 * [gspace.trivial_repr])`.
- Internal features are explicit `FieldType`s, not raw channel counts.
- Use `R2Conv` for every convolution.
- Use `R2Upsampling(mode="bilinear", scale_factor=2)` for decoder upsampling.
- Use `GeometricTensor` boundaries only inside model code. Training and loss code
  should receive ordinary tensors after model output extraction.

Representation plan:

1. Build a central field-type registry.
2. For SO(2), start with scalar fields, then add frequency-1 and
   frequency-2 fields.
3. Prevent accidental frequency or field mixing by constructing field types
   centrally.
4. Keep normalizations and nonlinearities selected from the field type, not from
   ad hoc channel counts.

Latent plan:

1. First version: scalar/trivial spatial latent fields with `(B, 16, 32, 32)`.
2. Second version: mixed-frequency latent fields with isotropic
   Gaussian sampling per nontrivial irrep/subfield.
3. Add a specific test: transform input, encode, compare transformed `mu` and
   valid `logvar` statistics, sample with controlled epsilon, and verify decoder
   behavior under the declared representation.

For nontrivial SO(2) latent fields:

- `mu` transforms as the declared irrep.
- `logvar` must not be an arbitrary per-coordinate tensor over vector components.
- Use one invariant variance scalar per irrep copy and spatial location.
- Sample `eps` isotropically inside each 2D irrep copy.
- KL must be computed using the matching isotropic Gaussian policy.

## Normalization Policy

Conservative default:

- No normalization in the first baseline and first equivariant model unless
  training stability requires it.

If normalization is needed:

- Non-equivariant baseline may use a `NormPolicy` abstraction, but the chosen
  policy must have an equivariant counterpart.
- Equivariant model should use `escnn` normalization modules such as
  `IIDBatchNorm2d`, `GNormBatchNorm`, `NormBatchNorm`, field norm, or no norm,
  depending on the representation.
- Raw GroupNorm is not allowed in the equivariant model. It may only be considered
  after a written proof/test that each group is a direct sum of compatible fields
  and the grouping operation commutes with the group action.

Practical note:

- `GNormBatchNorm` is more general but may be slower.
- `IIDBatchNorm2d` can be efficient when the field type contains repeated copies
  of larger reducible representations.
- BatchNorm statistics can interact with DDP and small batches. If batch size per
  rank becomes too small, prefer no norm or field/norm-based alternatives.

## Nonlinearity Policy

Componentwise nonlinearities are one of the easiest ways to quietly break an
`SO(2)` model. The baseline and equivariant model should therefore share a
field-aware activation policy rather than letting the baseline use arbitrary
componentwise SiLU everywhere.

Scalar fields:

- Ordinary scalar nonlinearities are valid.
- First choice: SiLU, `silu(x) = x * sigmoid(x)`.
- Alternative to test as an ablation: GELU-like `x * Phi(x)` or an
  erf-based smooth gate.

SO(2) vector/irrep fields:

- Do not apply SiLU/ReLU/GELU independently to vector components.
- For each 2D irrep copy `v = (u, w)`, compute an invariant radius statistic:

```text
r2 = u**2 + w**2
gate = sigmoid(a_i * r2 + b_i)
out = gate * v
```

- `a_i` and `b_i` are learned scalar parameters per irrep copy, or per frequency
  and copy. They must not be different for `u` and `w`.
- This radial gate is equivariant because `r2` is invariant and the same scalar
  gate multiplies both vector components.
- A richer ablation can replace `sigmoid(a_i * r2 + b_i)` with
  `silu(a_i * r2 + b_i)` or `erf(a_i * r2 + b_i)`, but negative gates should be
  treated carefully because they can flip vector phase by pi.
- Initialize `a_i` small and `b_i` near zero or mildly positive so the gate starts
  close to linear/pass-through rather than shutting down all vector fields.

Baseline comparability:

- The non-equivariant baseline uses full-mixing Conv2d over ordinary scalar
  channels.
- Every baseline channel uses the same gated scalar activation family as future
  equivariant scalar fields.
- Future nontrivial `SO(2)` vector/irrep fields use radial gates over invariant
  norms because ordinary componentwise scalar gates would break equivariance.
- This keeps the activation family conceptually aligned while letting the
  representation constraints, not fake vector groups in the baseline, define
  the difference between models.

Implementation note:

- Prefer an explicit `RadialGate` module over ad hoc tensor reshaping scattered
  through the model.
- The field schedule registry should distinguish ordinary baseline tensor
  channels from future `SO(2)` scalar fields and 2D irrep copies.
- Add tests that rotate a vector field and verify
  `activation(rho(theta) v) == rho(theta) activation(v)` within tolerance.

## Kernel And Frequency Policy

Because the target may use frequencies up to `L <= 2`, 3x3 kernels should not be
the default for the main 256x256 experiment.

Initial settings for 256x256:

- Stem kernel: 7x7.
- Hidden block kernels: 5x5 default.
- Early high-resolution blocks: 7x7 if compute allows.
- VAE heads: 5x5.
- Decoder post-upsampling kernels: 5x5.
- Avoid 1x1 unless the experiment explicitly studies it.

Initial settings for 32x32:

- Stem/hidden kernels: 3x3 or 5x5.
- Avoid 7x7 stride-2 stems on tiny inputs unless tested.

For `escnn.R2Conv`:

- Set `kernel_size` explicitly.
- Set or log `frequencies_cutoff`, `rings`, and `sigma` when deviating from
  defaults.
- For `L <= 2`, use a documented cutoff such as:
  `frequencies_cutoff=lambda r: 0 if r == 0 else min(2, 2 * r)`.
- Add a model summary that prints field types, kernel sizes, and frequency caps.

## Upsampling Validation Gate

Bilinear upsampling is the first decoder policy because it has a direct
`R2Upsampling(mode="bilinear")` counterpart and avoids PixelShuffle. It is still
not accepted on faith.

Required convention:

- Use integer uniform `scale_factor=2`.
- Use the same grid convention in baseline and equivariant code.
- Do not use arbitrary output `size` except in tests designed to measure the
  effect.

Required tests before full equivariant training:

- SO(2) decoder block equivariance over a fixed grid of sampled angles.
- Boundary-sensitive error report with and without a small border crop.
- Tolerance recorded in config or test docs.

Fallback if bilinear upsampling fails the SO(2) block test:

- Try `R2ConvTransposed` as a controlled decoder-upsample spike.
- Or redesign the decoder to keep upsampling outside the equivariant block only if
  the resulting approximation is explicitly documented and accepted as a limitation.

## Corruption And Augmentation Policy

- HED stain corruption is spatially pointwise and should commute with rotations
  and flips if the same per-image color parameters are used.
- Additive random noise does not commute sample-by-sample unless the transformed
  branch receives the transformed same noise. This matters for equivariance
  regularizers and equivariance tests.
- Rotation augmentation for the baseline must be logged explicitly. Do not compare
  a heavily rotation-augmented baseline against an equivariant model and then
  attribute all gains to equivariant layers.
- Keep corruptions identical between baseline and equivariant runs unless the
  ablation is explicitly about corruption.

Denoising contract:

- Training uses corrupted encoder input and clean decoder target.
- Validation reports both clean autoencoding and deterministic denoising.
- Equivariance tests use clean inputs by default.
- If an equivariance regularizer uses corrupted inputs, both transformed branches
  must receive transformed versions of the same corruption parameters/noise, not
  independently sampled corruption.
- All validation corruption seeds and parameter draws must be reproducible from
  the run config.

## Equivariance Evaluation Protocol

SO(2) protocol:

- Use a documented interpolation method, padding/cropping policy, and boundary
  mask.
- Report error with and without masked boundary pixels.
- Use fixed sampled angles for validation and store them in the run config.
- Treat transform-induced interpolation error as a baseline floor by measuring the
  same transform/inverse-transform roundtrip on raw inputs.
- Report dataset-level equivariance error in addition to 25-patch qualitative
  artifacts.
- Report latent-statistics equivariance for `mu` and valid `logvar`, not only
  reconstruction-level behavior.

## Repo Refactor Plan

Spec 0001 is the canonical layout for the first implementation. Target layout:

```text
src/eqvae/
  __init__.py
  config.py
  data/
    patch_shards.py
    synthetic.py
    splits.py
  corruption/
    stain.py
  models/
    __init__.py
    activations.py
    field_schedule.py
    non_equivariant_vae.py
  losses/
    vae.py
  metrics/
    reconstruction.py
  artifacts/
  checkpointing.py
  cli/
    smoke.py
    train.py
    benchmark_runtime.py
    select_fixed_patches.py
    evaluate.py
    artifacts.py
configs/spec0001/
  non_eq_vae_baseline.yaml
  non_eq_vae_debug_cpu.yaml
  non_eq_vae_kaggle_debug.yaml
  non_eq_vae_kaggle_runtime_benchmark.yaml
  ubc_ocean_masked_holdout_test.yaml
  fixed_25_validation_patches.yaml
docs/
  equivariant_vae_transition_plan.md
runs/
  README.md
```

Older names such as `ubc_patches.py`, `layer_schedule.py`,
`vae_non_equivariant.py`, or root-level `configs/non_eq_vae_ubc.yaml` are
superseded for spec 0001 unless a later spec intentionally reopens the layout.

Refactor rules:

- Kaggle notebooks should import from `src/eqvae` or serialize a generated script
  from repo code. They should not contain the canonical implementation.
- Training config should be serializable before GPU allocation, preserving the
  current traceability behavior.
- Data loading, model definition, loss, training loop, checkpointing, and
  evaluation artifact writing should be separate modules.
- Keep DDP, torch.compile, and Kaggle-specific constraints isolated in the
  training layer, not inside the model.
- Separate experiment outputs from datasets. Prefer `runs/` or external Kaggle
  output directories, not `data/processed` as a learned-output dumping ground.

## Milestones

### Phase -1: Behavior Inventory Before Refactor

- Extract the exact data contract from the Kaggle script:
  shape, dtype, range, binary header, CSV columns, split assumptions, and paths.
- Extract the exact training contract:
  optimizer, scheduler, AMP, DDP, compile, checkpoint fields, RNG state, and
  resume semantics.
- Extract the exact evaluation contract:
  reconstruction metrics, equivariance metrics, artifact paths, and checkpoint
  naming.
- Mark what belongs to the old FSQ experiment and must be removed.

Exit criteria:

- A written inventory exists at `docs/behavior_inventory_kaggle.md`.
- A one-batch old-script characterization run is reproducible, or the reason it
  cannot be run locally is documented.

### Phase 0: Lock The Specification

- Complete Phase -1 first.
- Input size is 256x256 continuation for spec 0001.
- Latent shape is reopened to the FSQ-successor spatial target
  `(B, 16, 32, 32)`.
- Confirm first implementation group: SO(2).
- Decide whether `O(2)` is a later ablation.
- Decide SO(2) representation schedule up to `L <= 2`.
- Decide whether normalization starts disabled.
- Use corrected Tellez-style stain-aware corruption plus per-image Gaussian noise
  and decide the remaining fairness budget details.

Exit criteria:

- Operation translation table has no "unknown" entries for the first baseline.
- The run config names the group, input size, latent shape, and layer schedule.
- `docs/specs/0001-translatable-normal-vae-baseline.md` is marked
  `locked / implementation-ready` with exact smoke, debug, resume, evaluator,
  and artifact-generation commands.

### Phase 1: Extract Reusable Infrastructure

- Move dataset loading for `.bin`/`.csv` UBC patch shards into `src/eqvae/data`.
- Move checkpointing, CSV logging, metrics, and reconstruction/equivariance
  artifact writers into reusable modules.
- Implement a shared evaluator/dashboard writer for SSIM, MAE, MSE, PSNR,
  mean, standard deviation, sample count `n`, metric boxplots, and
  training/evaluation dashboards.
- Keep current Kaggle paths configurable.
- Add a tiny local smoke dataset path or synthetic dataset path for tests.
- Add any missing package metadata and dependencies through `pyproject.toml` and
  `uv.lock`; keep Kaggle-only pip exports generated and documented if needed.

Exit criteria:

- A local CPU smoke test can instantiate data/model/loss without Kaggle.
- Kaggle launcher can still write config and run training.
- The shared evaluator can emit per-image metrics, summary tables, boxplots, and
  a dashboard placeholder from synthetic or smoke-test outputs.

### Phase 2: Implement Non-Equivariant Translatable VAE

- Implement the Conv2d VAE using only approved operations.
- Generate it from the shared layer schedule.
- Add beta KL schedule or fixed beta config.
- Keep stain/noise corruption, reconstruction loss, validation, checkpointing, and
  artifact writing.
- Remove FSQ-specific telemetry from this experiment.

Exit criteria:

- One debug run completes from start.
- One resume run completes from a midpoint checkpoint.
- Reconstruction, KL, SSIM, MAE, MSE, PSNR, sample count `n`, dataset-level
  equivariance error, boxplots, training/evaluation dashboard, and qualitative
  artifacts are logged.

### Phase 3: Implement escnn Feasibility Spike

- Build field type registry.
- Implement one encoder block, one downsample path, one decoder/upscale path, one
  VAE latent policy, and one output head.
- Test `R2Conv`, normalization, activation, upsampling, and output conversion.
- Explicitly test bilinear `R2Upsampling` with the chosen SO(2) sampled-angle
  protocol.

Exit criteria:

- `check_equivariance` passes for custom blocks where possible.
- End-to-end `SO(2)` rotation checks pass within documented tolerances.
  Reflection checks are only required for an explicit later `O(2)` ablation.
- Forward/backward and one optimizer step complete.

### Phase 4: Implement Full Equivariant VAE

- Implement the full shared layer schedule with `escnn` layer factories.
- Add frequency-1 and frequency-2 fields for SO(2).
- Add representation-aware latent sampling if using nontrivial latent fields.
- Match the baseline training protocol.
- Add equivariance regularizer only after the plain equivariant VAE trains.

Exit criteria:

- Full debug run completes.
- Resume from checkpoint works.
- Latent statistics equivariance tests pass.

### Phase 5: Experimental Comparison

- Train non-equivariant VAE baseline.
- Train equivariant VAE with matched data, schedule, and capacity notes.
- Export embeddings.
- Run linear probe or small MLP probe.
- Compare reconstruction, KL behavior, equivariance error, robustness to
  stain/noise, downstream metrics, parameter count, and compute.

Exit criteria:

- Results table and figures include SSIM, MAE, MSE, PSNR, mean, standard
  deviation, sample count `n`, metric boxplots, training/evaluation dashboards,
  fixed 25-patch artifacts, rotated-input artifacts, rotated-input versus
  transformed-latent grids, and EQ-VAE-style latent visualizations.
- The comparison reports seeds, tuning budget, augmentations, and compute.

## Test Plan

Unit tests:

- Shape checks for encoder, decoder, VAE output, KL, and loss.
- No banned ops in baseline model graph: FSQ, PixelShuffle, 1x1 conv, depthwise
  conv, nearest upsampling, grouped conv.
- Checkpoint save/load preserves epoch, step, optimizer, scheduler, scaler, RNG.
- Dataloader resume from `.5` checkpoint starts at the expected batch index.
- Field-type registry constructs exactly the expected representations.

Equivariance tests:

- For each equivariant block, run `check_equivariance` where available.
- For the full model, compare `model(g.x)` with `g.model(x)` for rotations and
  the chosen continuous `SO(2)` group. Reflection checks belong only to an
  explicit `O(2)` ablation.
- For the VAE encoder, compare transformed latent statistics, not random samples,
  unless epsilon is controlled.
- For decoder, test that transforming latent fields transforms reconstructions
  correctly.
- Report dataset-level equivariance error, not only 25-patch qualitative artifacts.

Training smoke tests:

- CPU synthetic batch: forward, loss, backward.
- Kaggle-only single-GPU tiny run if useful.
- DDP debug run on Kaggle with two ranks.
- Resume from midpoint and endpoint checkpoints.
- Export embeddings and run a tiny linear probe.

Adversarial checks:

- Does any transform use different random corruption/noise in the two equivariance
  branches?
- Does downsampling introduce unacceptable equivariance error?
- Does normalization mix channels from different irreps?
- Does latent variance accidentally transform as a vector field?
- Does a flattened latent destroy spatial equivariance?
- Does bilinear upsampling use nonuniform `size` instead of uniform
  `scale_factor`?
- Do metrics compare baseline and equivariant models under identical corruption
  and validation settings?
- Is the validation/test split leaking WSI, patient, or site information?
- Did the equivariant model receive more tuning trials than the baseline?

## Fairness Gates

Before claiming an equivariance result:

- Same train/validation/test split.
- Same input resolution and latent target, unless the experiment is explicitly an
  architecture ablation.
- Same optimizer family, schedule budget, corruption policy, and stopping rule.
- Same augmentation policy, or the difference is named as an ablation.
- Same number of seeds or a clear reason for fewer seeds.
- Same validation access and tuning budget.
- Report parameter count, approximate FLOPs/throughput, and wall-clock budget.
- Keep the test set sealed until final evaluation.
- The current pre-shuffled Kaggle dataset has train/validation files but no
  held-out test shard; generate and lock that shard before final claims.
- The held-out test shard should come from the WSIs with non-exhaustive
  supplemental masks; train/validation should remain on WSIs without those masks.

Capacity matching policy:

- Primary comparison is schedule-matched: the baseline and equivariant model use
  the same input size, downsampling depths, latent shape, kernel sizes, decoder
  structure, optimizer budget, and logging.
- The SO(2) equivariant model uses field multiplicities that correspond to the
  baseline channel-capacity schedule: future scalar, frequency-1, and
  frequency-2 multiplicities map to the total ordinary channel widths used by
  the baseline.
- Exact parameter equality is not required for the primary comparison because
  steerable kernels and weight sharing change the parameterization. The parameter
  gap must be reported.
- If parameter count differs by more than 25 percent, add a secondary
  parameter-matched or compute-matched ablation before making a strong performance
  claim.
- Tuning budget is matched by run count and validation access, not by whichever
  model is harder to stabilize.

## GitHub Issue Crosswalk

Issue #1, conferences:

- SIPAIM 2026 is the active conference target.
- Keep the conference issue updated with the SIPAIM page, submission dates, and
  the Overleaf/repo paper link.

Issue #2, baseline with ResNet18:

- The old ResNet18/FSQ baseline is historical and should not be the final
  comparison baseline.
- The replacement baseline is a normal continuous denoising VAE with operations
  chosen for SO(2) translation.
- Do not close until baseline metrics/plots have been produced for the new
  baseline or until the issue is explicitly re-scoped to historical FSQ results.

Issue #3, evaluation metrics:

- Required reconstruction metrics: SSIM, MAE, MSE, PSNR.
- Report mean, standard deviation, sample count `n`, and boxplots.
- Add KL metrics for the VAE.
- Add dataset-level SO(2) equivariance error for `mu`, valid `logvar`, latent
  samples with controlled epsilon, and reconstructions.

Issue #4, VAE validation:

- Generate original and reconstructed folders for 25 fixed patches.
- Generate rotated-input qualitative artifacts.
- For SO(2), include a fixed set of continuous angles in addition to 90/180/270
  degree visualizations.
- Add boxplots for the metrics over the evaluation set.
- Add latent visualization "a la EQ-VAE": continuous latent maps, transformed
  latent maps, and difference/error maps.
- Add `rotated_input_vs_latent_grid.*`: ground truth, rotated-input
  reconstruction, transformed-latent reconstruction, and error maps for the same
  patch/angle set.

Issue #5, SIPAIM 2026 writing:

- Keep a paper base in `paper/sipaim2026`.
- Use the IEEE conference manuscript style required by SIPAIM full papers.
- Maintain outline, related work/state of the art, methodology, experiments, and
  result placeholders in the repo.
- Add the relevant equivariant/steerable and histopathology references to
  `references.bib` as the literature review matures.

Issue #6, equivariant VAE validation:

- The target is SO(2).
- Implement the comparable SO(2) VAE after the non-equivariant translatable VAE.
- Use `escnn` where possible instead of hand-rolled steerable kernels.
- Explicitly test nonlinearities, normalization, upsampling, VAE sampling, and
  latent statistics for equivariance before running the full experiment.

## Implementation Defaults Pending Config Lock

These are active defaults or unresolved implementation choices that still need
to be encoded in configs. They are not invitations to reopen settled project
decisions such as the continuous `SO(2)` scope.

| Item | Default for now | Why it matters |
| --- | --- | --- |
| Input size | 256x256 continuation | Matches current Kaggle data pipeline. |
| Latent shape | `(B, 16, 32, 32)` | Preserves spatial coherence for the future continuous `SO(2)` comparison while removing FSQ quantization. |
| First implementation group | `SO(2)` via `rot2dOnR2(N=-1, maximum_frequency=2)` | This is the actual research target. |
| SO(2) hidden reps | Scalars plus frequency-1 and frequency-2 vector fields | Matches the `L <= 2` goal. |
| Latent reps | Scalar first, then irrep-aware vector latents | Avoids breaking VAE sampling on day one. |
| Normalization | None initially | Avoids representation-mixing mistakes and DDP stat issues. |
| Kernel size | 7x7 stem, 5x5 hidden, 5x5 heads for 256x256 | Gives steerable bases enough support for low frequencies. |
| Equivariance regularizer | Evaluation-only first | Separates architectural equivariance from training regularization. |
| Pointwise convs | Banned initially | Matches the intended translatable baseline constraint. |
| Upsampling | Bilinear scale factor + conv | Directly mirrors `R2Upsampling` and avoids PixelShuffle. |

## Immediate Next Tasks

1. Treat this document as the active checklist for the branch.
2. Finish relocking `docs/specs/0001-translatable-normal-vae-baseline.md`:
   parameter/FLOP count and final adversarial spec review. The runtime benchmark
   contract is written, but the benchmark result is a full-run gate after
   implementation.
3. Mark spec 0001 `locked / implementation-ready`, then add `src/eqvae` package
   skeleton.
4. Add configs that lock input size, latent shape, group, layer schedule,
   activation policy, runtime benchmark, and normalization.
5. Extract data/checkpoint/logging utilities from the Kaggle notebook.
6. Implement the non-equivariant translatable VAE.
7. Add banned-operation, activation-policy, shape, compile, and precision tests.
8. Replace the placeholder Kaggle debug kernel with the real CLI-managed
   launcher after local verification passes.
9. Resolve or explicitly baseline the strict Ruff/BasedPyright historical debt.
10. Run the short Kaggle runtime benchmark after explicit user permission, then
    choose single/dual GPU, AMP, compile, and batch size. Record this in
    `benchmark/selected_runtime.yaml` and the resolved full-run config.
11. Run the first 10-epoch Kaggle baseline only after benchmark selection and
    explicit user permission.
12. Implement the SO(2) `escnn` feasibility spike.
