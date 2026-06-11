# Repo Goal And Requirements Tracker

Status: active tracker
Last updated: 2026-06-10

This document keeps the repo horizon visible. It consolidates the research goal,
GitHub issue requirements, paper obligations, and evaluation artifacts so they
do not disappear inside issue comments or notebooks.

For the current handoff state and next concrete steps, read `CURRENT.md`.
For settled decisions, read `docs/decisions/README.md`.
For Kaggle dataset behavior, train/validation split verification, and the
masked-WSI holdout candidate list, read `docs/behavior_inventory_kaggle.md`.
For major requirement changes, use the adversarial review process in
`docs/agentic_review_workflow.md`.
For substantial implementation work, use the spec-driven workflow in
`docs/spec_driven_development.md` and write/update specs in `docs/specs/`.

Issue screenshots are part of the evidence. Read
`docs/issue_image_inventory.md` before changing evaluation requirements.

## Current Goal

Build and evaluate a comparable pair of histopathology patch VAEs:

1. A non-equivariant denoising VAE whose operations can be translated to the
   steerable model.
2. A continuous `SO(2)`-steerable denoising VAE implemented with repo-owned,
   compile-compatible SO(2) layers, using `escnn` as a reference.

The final claim should compare models that share:

- data pipeline and split policy;
- input size and normalization;
- latent target;
- downsampling/upsampling schedule;
- training budget and validation access;
- metric scripts and qualitative artifact protocol.

## Issue-Derived Requirement Tracker

| Source | Requirement | Acceptance artifact |
| --- | --- | --- |
| Issue #1, conferences | Keep SIPAIM 2026 as the active conference route and keep the paper links visible. | Issue comment includes SIPAIM/Overleaf/repo links; `paper/sipaim2026/README.md` has links. |
| Issue #2, baseline | Treat the old baseline as historical; produce metrics for the new comparable VAE baseline before closing or claiming baseline completion. | Baseline run config, checkpoint, metrics CSV, plots, and paper table entries. |
| Issue #3, metrics | Implement metric scripts for SSIM, MAE, MSE, and PSNR over test/eval images. | CSV or parquet with per-image metrics for each model. |
| Issue #3, metrics | Report mean, standard deviation, and sample count `n`. | Summary table in `paper/sipaim2026/tables/` and/or paper text. |
| Issue #3, metrics | Produce box-and-whisker plots for SSIM, MAE, MSE, and PSNR. | `paper/sipaim2026/figures/metrics_boxplots.*`, with `n` visible in caption or labels. |
| Issue #3/#4, attached dashboard images | Produce an analogous training/evaluation dashboard for each major run. | `paper/sipaim2026/figures/training_dashboard.*` with objective, reconstruction components, SSIM, PSNR, equivariance diagnostic, and learning-rate schedule. |
| Issue #3, metrics | Compare experiment 1 and experiment 2 using the same metric pipeline. | One shared evaluator, one comparison table, one comparison plot set. |
| Dataset gate, Kaggle | Keep final paper claims off the tuning validation set unless the sealed masked-WSI test shard has been generated and locked. | Test-shard dataset slug/mount path plus provenance from `docs/data/ubc_ocean_masked_holdout_ids.csv`. |
| Issue #4, VAE validation | Generate original and reconstructed artifacts for 25 fixed patches. | `paper/sipaim2026/figures/reconstructions_25.*` or linked folders/grids. |
| Issue #4, VAE validation | Generate rotated-input qualitative artifacts. | `paper/sipaim2026/figures/rotated_reconstructions_25.*` with fixed angles. |
| Issue #4, attached reconstruction image | Compare ground truth, rotated-input reconstruction, and transformed-latent reconstruction. | `paper/sipaim2026/figures/rotated_input_vs_latent_grid.*`, with error/difference maps if possible. |
| Issue #4, VAE validation | Produce boxplots for the 25-image validation subset if requested by advisor. | `paper/sipaim2026/figures/metrics_boxplots_25.*`, distinct from full eval plots. |
| Issue #4, VAE validation | Implement latent visualization "a la EQ-VAE". | `paper/sipaim2026/figures/latent_pca_eqvae_style.*`. |
| Issue #5, SIPAIM writing | Maintain SIPAIM paper base in IEEE conference style. | `paper/sipaim2026/main.tex`, `sipaim2026.pdf`, Overleaf project. |
| Issue #5, SIPAIM writing | Keep outline, related work, methodology, experiments, and result placeholders current. | Updated paper sections and tracked compiled PDF. |
| Issue #6, equivariant validation | Use continuous `SO(2)` as the target symmetry with a repo-owned implementation; use `escnn` as a reference, not the runtime dependency. | Config records continuous `SO(2)`, maximum frequency `L <= 2`, and the custom layer/downsample implementation choices. |
| Issue #6, equivariant validation | Validate nonlinearities, normalization, upsampling, VAE sampling, and latent statistics for equivariance before full runs. | Unit/block tests plus a small feasibility report. |

## Required Evaluation Artifacts

The paper should eventually have these figures/tables or explicit replacements:

- `paper/sipaim2026/figures/metrics_boxplots.*`
  - Boxplots for SSIM, MAE, MSE, PSNR.
  - Include sample count `n`.
- `paper/sipaim2026/figures/training_dashboard.*`
  - Objective curves.
  - Reconstruction component curves.
  - SSIM and PSNR curves.
  - Equivariance diagnostic.
  - Learning-rate schedule.
- `paper/sipaim2026/figures/reconstructions_25.*`
  - Fixed 25-patch original/reconstruction grid.
  - Same patch IDs for both models.
- `paper/sipaim2026/figures/rotated_reconstructions_25.*`
  - Fixed continuous angles and documented interpolation/padding policy.
- `paper/sipaim2026/figures/rotated_input_vs_latent_grid.*`
  - Ground truth, rotated-input reconstruction, and transformed-latent
    reconstruction for the same patch/angle set.
  - Include difference/error maps when possible.
- `paper/sipaim2026/figures/latent_pca_eqvae_style.*`
  - Top principal components of latent maps or latent representations.
  - Include baseline and `SO(2)` model outputs.
  - Include transformed latent maps and difference/error maps when available.
- `paper/sipaim2026/tables/metrics_summary.tex`
  - Mean, standard deviation, and `n` for each metric and model.
- `paper/sipaim2026/tables/equivariance_summary.tex`
  - Dataset-level equivariance errors for reconstructions and latent statistics.

## Metric Requirements

Reconstruction metrics:

- MSE;
- MAE;
- PSNR;
- SSIM.

VAE metrics:

- KL term;
- reconstruction term;
- beta schedule value;
- posterior statistics summaries for `mu` and valid `logvar`.

Equivariance metrics:

- reconstruction equivariance error under fixed sampled angles;
- latent `mu` equivariance error;
- valid `logvar` behavior for the chosen representation;
- sampled-latent equivariance with controlled or paired epsilon;
- raw image transform/inverse-transform roundtrip error as the interpolation
  floor.

All plots and tables must include sample count `n`.

## Fixed 25-Patch Protocol

Keep a deterministic set of 25 validation patches for qualitative artifacts.
Store enough metadata to reproduce it:

- patch IDs or file paths;
- WSI/patient/site identifiers when available;
- selection seed;
- preprocessing and corruption settings;
- fixed angles for rotated-input artifacts.

The 25-patch set is for qualitative and advisor-facing validation. Paper-level
metric claims should use the full validation/test set when feasible.

## EQ-VAE-Style Latent Visualization

The advisor explicitly requested a visualization similar to the EQ-VAE paper's
latent-space figures. For this repo, the required adaptation is:

1. collect latent maps/statistics for fixed validation images;
2. compute top principal components or another documented low-dimensional view;
3. show the baseline and `SO(2)` model with the same visualization pipeline;
4. include transformed inputs and transformed latent representations;
5. include difference/error maps where the representation type makes that
   meaningful;
6. report whether smoother or more structured latent geometry appears without
   compromising reconstruction.

Do not treat this as optional polish. It is an advisor-requested validation
artifact.

## Issue Image Handling

Issue images must be inspected before translating issue comments into plans.
The current inventory is in `docs/issue_image_inventory.md`.

- The dashboard screenshots in issues #3 and #4 define a required plot style for
  training/evaluation reporting.
- The qualitative reconstruction screenshot in issue #4 defines a required
  rotated-input versus transformed-latent comparison.
- The EQ-VAE screenshot in issue #4 defines the latent PCA/latent-map visual
  style to reproduce.

## Architecture Constraints That Affect Requirements

- Replace quantized bottlenecks with a normal continuous VAE.
- Use `mu`, `logvar`, and the reparameterization trick.
- Avoid sub-pixel/channel-to-space upsampling. Use bilinear upsampling plus
  convolution.
- Use kernels large enough for the `L <= 2` steerable basis, such as 5x5 or 7x7.
- Use Gaussian radial shells times real angular harmonics as the first
  repo-owned `SO(2)` kernel basis. Enforce zero center support for spatial
  angular frequencies `m > 0`; keep Bessel/Fourier-Bessel bases only as a future
  fallback/ablation.
- Use learned scalar gate parameters `a,b` in both the Conv2d baseline and
  `SO(2)` scalar/trivial fields to keep pointwise nonlinear expressivity
  comparable. Use radial gates for nontrivial `SO(2)` fields with
  `r = sqrt(||v||**2 + eps)` and an explicitly configured FP16-safe `eps`.
- Before full training, log a gate-health benchmark for learned gate parameters:
  saturation, `a,b` ranges, gradients/updates, and input/output RMS, so dead or
  saturated gates are caught before they can invalidate a full run.
- Before the first full Kaggle run, require runtime, dataloader-throughput,
  paired numerical, selected-runtime debug, checkpoint/resume, and tiny-overfit
  gates to pass on the selected configuration.
- Do not use a final `tanh` in the VAE output head. Use a zero-initialized final
  RGB convolution, train L1 on raw normalized output, and clamp only for
  SSIM/PSNR/images/artifacts outside the model forward path.
- Avoid arbitrary channel operations on `GeometricTensor` objects.
- Do not introduce a baseline layer unless the corresponding steerable layer is
  known or explicitly documented as a temporary non-comparable ablation.

## Completion Gates Before Paper Claims

Before claiming one model is better than the other:

1. Both models must use the same dataset split and evaluation images.
   Train/validation are locked to the verified pre-shuffled patch dataset; final
   paper claims require a sealed test shard generated from
   `docs/data/ubc_ocean_masked_holdout_ids.csv`.
2. Both models must use the same corruption protocol for denoising validation.
3. The metric script must be shared between models.
4. Boxplots and tables must include sample count `n`.
5. The fixed 25-patch qualitative artifacts must be regenerated for both models.
6. The EQ-VAE-style latent visualization must be regenerated for both models.
7. The training/evaluation dashboard must be regenerated for both models.
8. Equivariance tests must be reported for the `SO(2)` model.
9. Parameter count and compute differences must be reported.
10. `paper/sipaim2026/sipaim2026.pdf` must be refreshed.
11. Relevant GitHub issues must receive Spanish status updates.
