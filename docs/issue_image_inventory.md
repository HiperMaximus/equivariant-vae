# Issue Image Inventory

Status: active reference
Last updated: 2026-09-08

GitHub issue images are requirements evidence. Do not ignore them when updating
plans, evaluation scripts, paper figures, or issue comments.

## Summary

| Issue | Image | Visual content | Requirement impact |
| --- | --- | --- | --- |
| #3 | `https://github.com/user-attachments/assets/ac6d07ff-81c1-455e-b986-786da5a58649` | Six-panel training/evaluation dashboard: composite objective, Charbonnier approximation, SSIM offset, PSNR, bottleneck spatial equivariance over 25 samples, learning-rate schedule. | The accepted report preserves an analogous dashboard rather than only final summary metrics. |
| #4 | `https://github.com/user-attachments/assets/6ffcfa4e-bdeb-4b09-9697-9d97872a2482` | Same six-panel dashboard as issue #3. | Cross-confirms the dashboard requirement for validation reporting. |
| #4 | `https://github.com/user-attachments/assets/9c6df0e6-6d73-4f80-9c05-34431b600f3e` | Qualitative grid with columns `Ground Truth`, `Rotated Input Reconstruction`, `Rotated Latent Reconstruction` and rows 0, 90, 180, 270 degrees. Rotated latent reconstructions visibly diverge for nonzero rotations. | Need fixed-angle reconstruction grids comparing image-space rotation and latent-space transformation, plus error/difference maps when possible. |
| #4 | `https://github.com/user-attachments/assets/cf38dd80-0467-4f6b-bf24-4e4fccea1320` | EQ-VAE-style PCA-color latent visualization comparing baseline VAE latents and improved/equivariant latents. The target visual pattern is smoother, more structured latent maps without losing reconstruction content. | Need side-by-side latent PCA/latent-map visualizations for baseline and `SO(2)` model. |
| #6 | `https://github.com/user-attachments/assets/193001bf-d76d-4e29-883c-2bb4b6609d1b` | Dense 0°--359° latent-orbit grid for all 25 fixed validation patches, paired normal/`SO(2)`, with population summaries for local linearity, step-size CV, and PCA planarity. | Superseded provenance only: its trajectory mixed opposite conventions at cardinal angles. The authorized corrected comment removed this embed and withdrew its 25/25, step-CV, PCA-orbit and continuous-F1 claims; the remote asset itself remains historical. |
| #6 | `https://github.com/user-attachments/assets/e9296611-e2e2-47ae-9c89-3a5e37b2553e` | Supervised-development chart with MIL validation macro-F1, online train versus validation cross-entropy, and tissue validation macro-F1 across label budgets. | Show train context next to validation while labeling online train loss as an optimization diagnostic, not fixed-checkpoint train performance or sealed-test evidence. |

## Current Professor Comment

The authorized final Spanish status comment is
`https://github.com/HiperMaximus/equivariant-vae/issues/6#issuecomment-5578529496`.
It remains on the open issue and was replaced with the corrected Spec 0050
conclusion. Its 26-page/20-figure local-report snapshot withdrew the
mixed-sign figures, found no demonstrated shared action, and left
factorization unresolved. The later repository report is the reviewed
29-page/22-figure artifact named in `CURRENT.md`; it was not republished to
the issue. The comment intentionally no longer embeds the two defective
dense-orbit images and preserves the ten verified attachments unaffected by
the defect; the inventory retains the old all-25 URL only as provenance.
Further issue mutation requires a new explicit request.

## Image-Derived Artifact Contract

The accepted report implements these requirements; preserve the same semantics
when transferring results elsewhere:

1. Training/evaluation dashboard.
   - Include train/validation objective curves.
   - Include reconstruction component curves.
   - Include SSIM and PSNR curves.
   - Include learning-rate schedule.
   - Include an equivariance diagnostic curve for the fixed qualitative subset
     or a dataset-level replacement.
2. Metric boxplots.
   - SSIM, MAE, MSE, PSNR.
   - Include sample count `n`.
   - Use the same evaluator for baseline and `SO(2)` model.
3. Fixed-angle qualitative reconstruction grid.
   - Ground truth.
   - Rotated-input reconstruction.
   - Transformed-latent or latent-action reconstruction.
   - Fixed angles, including continuous angles for the `SO(2)` model.
   - Include error/difference maps when possible.
4. EQ-VAE-style latent visualization.
   - Use the same PCA/color projection or documented alternative for both
     models.
   - Show baseline and `SO(2)` model side by side.
   - Show transformed latent maps and difference/error maps where meaningful.
   - State whether the `SO(2)` model produces smoother or more structured
     latents without damaging reconstruction.
5. Dense rotation-orbit population view.
   - Show all 25 fixed validation patches for both models at one-degree steps.
   - Keep local-linearity, traversal-uniformity, and PCA-planarity conclusions
     separate; do not treat PCA appearance as a confirmatory metric.
6. Supervised train/validation context.
   - Pair checkpoint-selection validation curves with available train telemetry.
   - Distinguish online train loss from evaluation of a fixed checkpoint and
     keep both separate from sealed-test results.

## Documentation Hygiene

- When an issue image contradicts a text summary, update the docs to reflect the
  inspected image.
- Do not keep stale/bad/incorrect information in README files, plans, memories,
  or issue trackers. Delete or replace it with the current source of truth.
