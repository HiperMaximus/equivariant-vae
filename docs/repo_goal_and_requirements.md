# Goal And Evaluation Requirements

Status: complete for the professor report
Last updated: 2026-09-08

## Goal

Compare a normal denoising VAE with a matched repo-owned continuous-`SO(2)` VAE
on reconstruction, rotation behavior and downstream histopathology utility.
Both models share data, input/latent shapes, training budget, validation access
and evaluation code.

The final advisor-facing synthesis is the 20-page, 15-figure pair
`reports/professor/informe_final_experimento_eqvae.{docx,pdf}`. Exact results
and hashes live in `CURRENT.md`.

## Professor Requirements

| Source | Required evidence | Accepted location |
| --- | --- | --- |
| Issue #2 | Comparable normal-VAE baseline results | Frozen normal checkpoint, Spec 0044 and final report |
| Issue #3 | SSIM, MAE, MSE and PSNR; mean, SD, `n`; boxplots; training/evaluation dashboard | `runs/local/professor_metrics_v1` and report Figures 2–4 |
| Issue #4 | Fixed 25 originals/reconstructions; rotated-input versus transformed-latent view; EQ-VAE-style latent PCA visualization | `runs/local/professor_metrics_v1`, `runs/local/frozen_vae_rotation_orbits` and report Figures 4–9 |
| Issue #6 | Repeat evaluation for continuous `SO(2)`; test downstream WSI/tissue utility | Specs 0038–0048 and report Figures 8–14 |
| Issue #6 follow-up | Verify the 0°–359° orbit across all 25 fixed patches at one-degree resolution | Legacy Spec 0038 artifacts are superseded by a mixed-sign rotation defect; corrected validation is owned by Spec 0050. |
| Issue #6 follow-up | Show train context with WSI validation curves | Report Figure 10; online train CE is labelled optimization telemetry |
| Issue #6 follow-up | Consider WSI patch attribution | Deferred: accepted outputs lack patch-level gates; instrumented inference is required |

The final Spanish status comment is
`https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496`.
The issue remains open.

## Evaluation Populations

Never merge or relabel these populations:

| Population | Unit and use |
| --- | --- |
| Fixed validation 25 | Qualitative reconstruction, rotation and latent diagnostics |
| Full reconstruction test | 67,138 patches clustered in 23 sealed-test WSIs |
| WSI diagnosis test | 23 complete WSI bags; support CC/EC/HGSC/LGSC/MC = 5/6/8/2/2 |
| Tissue test | 31,572 patches from 23 WSIs at five nested label budgets |
| Development train/validation | Optimization and checkpoint selection only |

Fixed-25 evidence is not sealed-test evidence. Patch-level dispersion is not
uncertainty for a WSI-level claim.

## Metrics

Reconstruction:

- MAE, MSE, per-image PSNR and SSIM;
- mean, population standard deviation and sample count `n`;
- paired model differences with direction stated;
- WSI-cluster bootstrap intervals for the full sealed reconstruction test.

VAE training:

- reconstruction and KL components;
- beta schedule;
- posterior `mu` and valid `logvar` summaries;
- clean and deterministic-denoising validation views;
- online train telemetry explicitly separated from fixed-checkpoint evaluation.

Rotation/equivariance:

- image transform/inverse-transform roundtrip floor;
- reconstruction and posterior-`mu` rotation behavior;
- valid `logvar` and controlled-epsilon sampling behavior;
- transformed-latent decoding;
- local-linearity, step-size CV and PCA-planarity proxies reported separately.

Downstream:

- WSI macro-F1, accuracy, balanced accuracy, per-class F1 and confusion matrix;
- tissue macro-F1 and per-class F1 across label budgets;
- paired bootstrap intervals with the correct sampling unit;
- class and WSI support next to claims.

## Required Visuals

The accepted report contains:

1. training/evaluation dashboard;
2. fixed-25 reconstruction grid;
3. reconstruction metric boxplots;
4. rotated-input versus transformed-latent comparison;
5. selected-patch 0°–359° orbit;
6. all-25 one-degree orbit sheets;
7. common-scale spatial latent PCA/coherence views;
8. WSI development curves with online train-loss context;
9. sealed WSI metrics, confusion matrices and per-class F1;
10. tissue label-efficiency and per-class F1 plots.

`docs/issue_image_inventory.md` records the inspected screenshots and final
GitHub attachment URLs.

## Claim Gates

- Primary reconstruction: normal-minus-`SO(2)` MAE with WSI-cluster bootstrap.
- WSI diagnosis: one seed and 23 WSIs; a higher point estimate is not proof of
  superiority when the interval crosses zero.
- Tissue: six-contrast simultaneous intervals; only the 500-label contrast
  excludes zero.
- Rotation: the former 25/25 dense interpretation is superseded by a mixed-sign
  defect. The corrected one-degree local-linearity direction misses the fixed
  10% gate and reverses at five degrees; neither a shared reduced action nor
  local content--pose factorization was demonstrated. Exact-quarter residuals
  remain valid and do not favor `SO(2)`.
- PCA colors and individual examples are diagnostics, not performance metrics.
- No universal winner is established.
- No attention heatmap may be fabricated from logits or graph coordinates.

## Current Boundary

The sealed evaluation and corrected fixed-validation rotation audit are
complete. The professor report contains the reviewed Spec 0050 result. Paper,
thesis, Overleaf, public derived-data release, further issue mutation and WSI
attribution require separate explicit scope.
