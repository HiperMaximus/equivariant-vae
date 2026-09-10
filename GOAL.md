# Repository Goal

## North Star

Measure whether continuous rotation-equivariant structure improves
histopathology patch representation learning under a fair comparison between:

1. a normal non-equivariant denoising VAE; and
2. a repo-owned continuous-`SO(2)` steerable denoising VAE.

Both models share the data, 256x256 input contract, `float32[16,32,32]`
Gaussian latent target, macro-architecture schedule, optimization budget,
validation access and evaluation pipelines. The `SO(2)` model uses an
equal-or-smaller learned-parameter budget.

## Current Outcome

The two 60,000-update VAEs, frozen latent stores, supervised WSI and tissue
comparisons, and sealed reconstruction test remain complete and unchanged.
Spec 0050 completed the validation-only correction and audit of a mixed-sign
continuous-rotation defect and rebuilt the professor report from the reviewed
corrected evidence.

The evidence is mixed:

- normal has a small descriptive reconstruction advantage;
- `SO(2)` has favorable downstream point estimates without a general primary
  test advantage;
- the corrected one-degree direction is descriptively favorable to `SO(2)` but
  misses the fixed effect-size gate and reverses at five degrees;
- no low-dimensional shared action, clean internal-F1 transformation or local
  content--pose factorization was demonstrated; the raw-`mu` exact-quarter
  control remains valid and does not favor `SO(2)`;
- separately, the SO(2) decoder realizes the prescribed spatial C4 action at
  exact 90/180/270-degree rotations only, not continuous SO(2), encoder
  equivariance, or complete D4/O(2).

The current Spanish report is
`reports/professor/informe_final_experimento_eqvae.{docx,pdf}`. Its sealed and
supervised sections remain accepted and its rotation section now supersedes the
defective dense figures with the reviewed Spec 0050 results. `CURRENT.md` owns
exact state and authorization.

## Scientific Contract

- Symmetry target: continuous `SO(2)`, not a discrete rotation group.
- Bottleneck: Gaussian VAE using `mu`, valid `logvar` and reparameterization;
  no FSQ, codebooks, rounding or discrete latent indices.
- Output: zero-initialized raw RGB convolution; clamp only for image-domain
  metrics and artifacts, not in model forward.
- Resampling: fixed anti-aliased downsampling and bilinear-upsample-plus-conv;
  no PixelShuffle or nearest-neighbor equivariant upsampling.
- Equivariant kernels: Gaussian radial shells times real angular harmonics,
  zero center support for spatial frequencies `m > 0`, selected F0/F1 layout.
- Normalization and nonlinearities: ordinary baseline operations paired with
  field-aware `SO(2)` counterparts; learned gates are monitored for saturation
  and updates.
- Fairness: matched data/splits, latent shape, training budget, validation
  access, metric code and qualitative protocol.
- Test discipline: sealed results never drive tuning, checkpoint selection,
  retraining or retries.

## Required Evaluation Surface

- Reconstruction: MAE, MSE, PSNR and SSIM with mean, population SD and `n`.
- Uncertainty: paired WSI-cluster bootstrap intervals for WSI-level claims;
  descriptive patch dispersion must not be presented as inferential
  uncertainty.
- Visuals: training dashboard, reconstruction boxplots, fixed-25 grids,
  rotated-input versus transformed-latent comparisons, EQ-VAE-style spatial
  PCA views and all-25 one-degree latent orbits.
- Downstream: WSI diagnosis and tissue label efficiency with train context,
  validation/checkpoint-selection separation, sealed-test metrics, class
  support and limitations.
- Attribution: no WSI heatmap unless patch-level values are recovered through
  a separately scoped instrumented pass.

## Sources Of Truth

- Operational state and exact evidence: `CURRENT.md`.
- Repository rules: `AGENTS.md`.
- Requirements and metrics: `docs/repo_goal_and_requirements.md`.
- Architecture contract: `docs/equivariant_vae_transition_plan.md`.
- Data/source contract: `docs/behavior_inventory_kaggle.md`.
- Remote execution: `docs/kaggle_cli_workflow.md`.
- Spec status and detailed contracts: `docs/specs/README.md` and linked specs.
- Settled design choices: `docs/decisions/README.md`.
- Professor image requirements: `docs/issue_image_inventory.md`.

## Boundaries

- SIPAIM 2026 was not submitted; do not present it as an active venue.
- `paper/sipaim2026` remains the working manuscript.
- The thesis repository is separate.
- Paper, thesis, Overleaf, GitHub mutations, commits/pushes, public derived-data
  release and new inference require explicit scope.
