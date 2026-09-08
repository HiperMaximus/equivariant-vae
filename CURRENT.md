# Current Repository Status

Last updated: 2026-09-08

## Handoff

The experiment and professor-facing synthesis are complete. No Kaggle job or
monitor is active. Presentation is ready; paper, thesis, Overleaf, commits,
pushes, public data release and new WSI attribution inference require separate
scope.

Final deliverables:

- `reports/professor/informe_final_experimento_eqvae.docx`
  - SHA-256: `0ee1a866ed98dda1478a0b6da3568f5990f8a1c7af8319157107573369d58da1`.
- `reports/professor/informe_final_experimento_eqvae.pdf`
  - SHA-256: `3524c206d484b01591ac025936c44a986449ffeb6ac70de4876a1434a9653451`.
- 20 Letter pages and 15 inline figures.
- Final Spanish issue update:
  `https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496`.
  Issue #6 remains open and the comment embeds 12 verified attachments.

The report passed full-page visual inspection, image and accessibility audits,
builder checks and two independent claim/layout reviews with no P0/P1/P2.

## Frozen Models

Both denoising VAEs use the matched `float32[16,32,32]` Gaussian posterior-`mu`
target and completed 60,000 optimizer updates.

- Normal VAE:
  `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`
  - SHA-256: `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075`.
  - Final clean/denoising L1: `0.059252/0.062364`.
  - Final clean/denoising SSIM: `0.733635/0.725689`.
  - Preserve the accepted single physical-update legacy exception.
- Continuous-`SO(2)` VAE:
  `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`
  - SHA-256: `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7`.
  - Final clean/denoising L1: `0.0611158/0.0639382`.
  - Final clean/denoising SSIM: `0.719666/0.711492`.
  - Preserve the accepted non-clean 66/68 gate-health result: one decoder and
    five encoder F1 channels saturated open while retaining finite positive
    gradients and updates.

The checkpoints are frozen. Do not retrain, tune or alter them.

## Frozen Data Contract

- Unsupervised development source:
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
- Masked-WSI cohort: 152 WSIs with a frozen shared `106/23/23`
  train/validation/test split.
- Full foreground population: 1,750,221 Otsu-selected patches represented for
  both models.
- Tissue test: 31,572 high-purity labelled patches from 23 test WSIs.
- Reconstruction test: 67,138 cancer-task patches from the same 23 sealed-test
  WSIs.
- Physical latent stores remain remote and immutable; logical task/split views
  reference them without copying binary payloads.
- Supplemental masks are non-exhaustive. Black/unannotated regions are not a
  negative tissue class.

Specs 0017–0025 and their artifact manifests own membership, coordinate,
storage and provenance details. Never use sealed-test results for tuning,
selection or automatic retries.

## Accepted Results

### Full-Test Reconstruction

Evidence: `runs/local/vae_test_reconstruction_scored_v1`.

- Primary pooled MAE, normal/`SO(2)`: `0.0628721/0.0642299`.
- Paired normal-minus-`SO(2)` MAE: `-0.00135779`.
- Prespecified 10,000-draw WSI-cluster bootstrap 95% interval:
  `[-0.00221281,+0.00004591]`.
- Secondary normal/`SO(2)` results:
  - MSE: `0.00799258/0.00827155`;
  - PSNR: `27.92255/27.65827` dB;
  - SSIM: `0.757498/0.743584`.

The point estimate favors normal reconstruction, but the primary interval
includes zero. Secondary endpoints and diagnosis strata are exploratory.

### WSI Diagnosis

Evidence: `runs/local/ubc_ocean_mil_test_scored_v1`.

- Test support CC/EC/HGSC/LGSC/MC: `5/6/8/2/2` WSIs.
- Macro-F1, normal/`SO(2)`: `0.38880/0.48333`.
- Paired normal-minus-`SO(2)` difference: `-0.09454`.
- WSI-bootstrap 95% interval: `[-0.20712,0.04968]`.

The `SO(2)` point estimate is higher, but the interval crosses zero. This is one
classifier seed on 23 WSIs and does not establish a general advantage.

### Tissue Label Efficiency

Evidence: `runs/local/tissue_test_scored_v1`.

Normal/`SO(2)` macro-F1 at 250/500/1000/2500/5671 labels per class:

- `0.5078/0.5228`;
- `0.5109/0.5752`;
- `0.5822/0.5979`;
- `0.7007/0.7136`;
- `0.7657/0.7102`.

Only the 500-label contrast excludes zero under the prespecified six-contrast
simultaneous 95% intervals and favors `SO(2)`. Normal/`SO(2)` log-budget AULC is
`0.6151/0.6314`; paired difference `-0.0163 [-0.0728,0.0402]`. Necrosis occurs
in only five test WSIs and uncertainty excludes training-seed variation.

### Latent Rotation And Spatial Diagnostics

Evidence: `runs/local/frozen_vae_rotation_orbits` and
`runs/local/professor_metrics_v1`.

The all-25 dense orbit uses exactly 25 fixed validation patches and all 360
integer angles. `SO(2)` has:

- lower raw-space local-linearity ratio in `25/25` pairs, median
  `0.983993` versus normal `1.033922`;
- lower step-size CV in `0/25`, median `0.204670` versus normal `0.189132`;
- higher two-PC explained variance in `9/25`, median `0.038098` versus normal
  `0.039123`.

This supports only greater one-degree local smoothness for `SO(2)`. It does not
support more uniform traversal, greater PCA planarity, exact equivariance or
universal superiority. Fixed-25 reconstruction and PCA views are validation
evidence, not sealed-test metrics.

## Interpretation Boundary

The integrated result is mixed:

- normal is slightly better descriptively on reconstruction;
- `SO(2)` has favorable point estimates in WSI diagnosis and several low-label
  tissue budgets;
- `SO(2)` is locally smoother at one-degree resolution across all 25 fixed
  validation patches;
- no primary test establishes a universal winner.

No WSI attention map exists in the accepted outputs because patch-level gates
were not retained. A defensible map requires a separately authorized,
instrumented frozen-checkpoint inference pass and spatial projection. Raw
attention must not be presented automatically as faithful class attribution.

## Current Operational State

- All MIL, tissue, reconstruction-test and dense-orbit one-shot authorities are
  consumed.
- No launch, retry, retraining, tuning, checkpoint change or remote mutation is
  authorized.
- SIPAIM 2026 was not submitted. `paper/sipaim2026` is only the working
  manuscript and was not changed for the final report.
- The worktree contains substantial intentional modified and untracked research
  files. Preserve them; do not reset or clean them.
- Live repository documentation contains current contracts and outcomes only;
  do not restore execution diaries or completed follow-up lists.
- The repo-wide Python gate is currently blocked only by 23 lint findings in
  packaged probe files; focused report checks pass.

## Next Authorized Boundary

Present the final report. Any paper, thesis, Overleaf, commit/push, public-data
release, issue mutation or WSI attribution work requires a new explicit request.
