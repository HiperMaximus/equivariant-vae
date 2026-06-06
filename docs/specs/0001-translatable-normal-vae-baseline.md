# Spec 0001: Translatable Normal VAE Baseline

Status: draft active spec
Implementation readiness: not ready
Owner/workstream: comparable non-equivariant VAE baseline
Last updated: 2026-06-06

## Purpose

Replace the historical FSQ autoencoder experiment with a normal denoising VAE
baseline whose operations can be translated to the future continuous `SO(2)`
steerable model.

This is the first implementation target before building the full `escnn` path.

Do not implement this spec yet. It must first be locked as
`implementation-ready` after the Kaggle behavior inventory and exact
verification commands exist.

## Non-Goals

- No FSQ, vector quantization, codebooks, discrete indices, or quantized latent
  telemetry.
- No final performance claims against the steerable model.
- No thesis repo update.
- No reflection or other symmetry ablation.
- No arbitrary baseline layer unless its steerable counterpart is already known
  or explicitly documented as a temporary ablation.

## Inputs And Data Contract

Default input contract until the spec is locked:

- dataset: UBC-OCEAN histopathology patches from the current Kaggle pipeline;
- source behavior inventory: `docs/behavior_inventory_kaggle.md`, derived from
  `kaggle/train_runs` and `kaggle/dataset_generation`;
- image size: 256x256 RGB unless explicitly changed in config;
- normalization: `[-1, 1]`;
- training input: corrupted patch `x_in = corrupt(x_clean)`;
- target: clean patch `x_clean`;
- split policy: WSI/patient/site-level split where metadata allows, never
  patch-level leakage;
- latent target: spatial Gaussian latent `(B, 24, 4, 4)` unless an ablation spec
  says otherwise.

The behavior inventory for the current Kaggle scripts has been written and must
be used before copying code into reusable modules.

Required inventory artifact:

```text
docs/behavior_inventory_kaggle.md
```

## Architecture Contract

The baseline must be generated from a layer schedule that the equivariant model
can reuse.

Allowed first-run operations:

- odd square `Conv2d`, preferably 5x5 or 7x7;
- strided odd-kernel convolution for downsampling;
- bilinear upsampling plus convolution for upsampling;
- residual adds only when shapes and semantics match;
- spatial Gaussian VAE latent map;
- scalar/radial activation policy that can be mirrored in the steerable model.

Banned first-run operations:

- FSQ or any vector-quantized bottleneck;
- PixelShuffle or sub-pixel convolution;
- nearest-neighbor upsampling in the comparable path;
- 1x1 pointwise convolutions unless a future spec explicitly permits them;
- depthwise/grouped/MBConv/squeeze-excite/channel-attention operations;
- arbitrary flattening or channel slicing that cannot be mirrored for
  `GeometricTensor` fields.
- FSQ-era resume sources or discrete-latent artifact requirements.

## Objective Contract

Use a normal denoising VAE:

```text
z = mu + exp(0.5 * logvar) * eps
recon_loss = mean((x_hat - x_clean) ** 2)
kl_element = -0.5 * (1 + logvar - mu ** 2 - exp(logvar))
kl_loss = mean(kl_element)
loss = recon_loss + beta * kl_loss
```

First beta policy:

- linear warmup from 0 to 1 over the first 10 percent of training, unless config
  changes this explicitly.

Log SSIM, MAE, MSE, PSNR, KL, reconstruction loss, posterior statistics, and
learning rate. SSIM is a metric for the first run, not a training loss.

## Required Implementation Artifacts

- reusable data module for patch shards and metadata;
- reusable config for input size, latent shape, layer schedule, corruption,
  optimizer, and beta schedule;
- model factory for the non-equivariant translatable VAE;
- checkpoint save/resume utilities;
- per-image evaluator for SSIM, MAE, MSE, PSNR with sample count `n`;
- artifact writers for boxplots, training/evaluation dashboards, fixed
  25-patch reconstructions, rotated-input artifacts, rotated-input versus
  transformed-latent grids, and EQ-VAE-style latent visualizations.

## Acceptance Criteria

The baseline is not complete until:

1. `docs/behavior_inventory_kaggle.md` records the current Kaggle data,
   training, resume, evaluation, and artifact behavior;
2. this spec is updated from draft active to implementation-ready before coding;
3. a CPU smoke test can instantiate data, model, loss, and evaluator;
4. a debug training run completes from start;
5. a resume run completes from a midpoint checkpoint;
6. metrics include per-image SSIM, MAE, MSE, PSNR and summary mean/std/`n`;
7. boxplots and a training/evaluation dashboard are produced;
8. the fixed 25-patch qualitative artifact protocol is implemented;
9. rotated-input qualitative artifacts use fixed continuous angles;
10. `rotated_input_vs_latent_grid.*` can be produced for the same patch/angle set;
11. banned-operation checks pass;
12. `./scripts/python_quality.sh` passes, or any failure is limited to the
    documented historical Ruff/BasedPyright debt with unchanged counts and no
    new debt outside `main.py` / historical exploratory `src/nn` files;
13. the implementation spec and `CURRENT.md` are updated with any changed
    contract details.

## Verification Commands

These are placeholders until the package/test structure is implemented:

```bash
./scripts/agent_preflight.sh
./scripts/python_quality.sh
```

Once the code exists, add exact smoke-test, evaluator, and artifact-generation
commands here.

## Implementation Blockers

- Exact local CPU smoke-test command is not defined.
- Exact debug training command is not defined.
- Exact resume command is not defined.
- Exact evaluator/dashboard/artifact-generation commands are not defined.
- Final input size and split policy are not locked.
- Normalization and scalar/radial nonlinearity policy are not locked.
- The strict Python quality gate still has historical exploratory-code debt.

## Known Risks

- Starting implementation from this draft could copy FSQ-era behavior into the
  new VAE path.
- Placeholder commands could let future agents call the spec "done" without a
  runnable local CPU workflow.
- A too-flexible baseline could become stronger or less constrained than the
  future continuous `SO(2)` model, making the comparison unfair.

## Adversarial Checks

- Does any operation violate the future continuous `SO(2)` translation path?
- Does the VAE objective accidentally omit KL or log invalid `logvar` behavior?
- Does corruption randomness differ between comparison branches?
- Do metric scripts include `n` and run on the same evaluation images?
- Do qualitative artifacts use the same 25 patch IDs for both future models?
- Does any split leak WSI, patient, or site information?
- Does the baseline receive more tuning than the future equivariant model?

## Open Questions Before Full Runs

- Lock final input size: 256x256 continuation or 32x32 thesis-return.
- Lock split metadata source and leakage checks.
- Lock normalization policy for the comparable baseline and steerable model.
- Lock activation/radial nonlinearity policy for vector-like lanes.
- Decide whether normalization starts disabled or uses a tested steerable-safe
  equivalent.

## Related Files

- `GOAL.md`
- `docs/equivariant_vae_transition_plan.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/decisions/0001-continuous-so2-scope.md`
- `docs/decisions/0002-normal-vae-baseline.md`
