# Issue Image Inventory

Status: active reference
Last updated: 2026-06-05

GitHub issue images are requirements evidence. Do not ignore them when updating
plans, evaluation scripts, paper figures, or issue comments.

## Summary

| Issue | Image | Visual content | Requirement impact |
| --- | --- | --- | --- |
| #3 | `https://github.com/user-attachments/assets/ac6d07ff-81c1-455e-b986-786da5a58649` | Six-panel training/evaluation dashboard: composite objective, Charbonnier approximation, SSIM offset, PSNR, bottleneck spatial equivariance over 25 samples, learning-rate schedule. | Future runs need an analogous training dashboard, not only final summary metrics. |
| #4 | `https://github.com/user-attachments/assets/6ffcfa4e-bdeb-4b09-9697-9d97872a2482` | Same six-panel dashboard as issue #3. | Cross-confirms the dashboard requirement for validation reporting. |
| #4 | `https://github.com/user-attachments/assets/9c6df0e6-6d73-4f80-9c05-34431b600f3e` | Qualitative grid with columns `Ground Truth`, `Rotated Input Reconstruction`, `Rotated Latent Reconstruction` and rows 0, 90, 180, 270 degrees. Rotated latent reconstructions visibly diverge for nonzero rotations. | Need fixed-angle reconstruction grids comparing image-space rotation and latent-space transformation, plus error/difference maps when possible. |
| #4 | `https://github.com/user-attachments/assets/cf38dd80-0467-4f6b-bf24-4e4fccea1320` | EQ-VAE-style PCA-color latent visualization comparing baseline VAE latents and improved/equivariant latents. The target visual pattern is smoother, more structured latent maps without losing reconstruction content. | Need side-by-side latent PCA/latent-map visualizations for baseline and `SO(2)` model. |

## Image-Derived Artifact Requirements

The current experiment/paper workflow must produce these artifacts or explicitly
justify replacements:

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

## Documentation Hygiene

- When an issue image contradicts a text summary, update the docs to reflect the
  inspected image.
- Do not keep stale/bad/incorrect information in README files, plans, memories,
  or issue trackers. Delete or replace it with the current source of truth.
