# Spec 0010: Fixed-25 Evaluation Protocol

Status: implemented and verified for both frozen models
Implementation readiness: closed; selector and artifacts are immutable
Owner/workstream: qualitative reconstruction and rotation evaluation
Last updated: 2026-09-08

## Purpose

Provide one deterministic validation subset for matched qualitative and
embedding-space comparison of the normal and continuous-`SO(2)` VAEs.

## Selector Contract

- Exactly 25 unique validation patches.
- Same ordered patch identities for both models.
- Manifest records source locator, WSI/row identity, CRC/integrity status,
  selection seed and hashes.
- Selector generation is label-independent and never uses sealed-test data.
- Canonical remote selector guard token: `fixed25_selector_kernel_ready`.
  The selector already exists; this token is not permission to relaunch.

## Evaluation Contract

At an accepted checkpoint or post-hoc frozen pass:

- decode clean posterior means for original/reconstruction grids;
- compute fixed-25 MAE, MSE, PSNR and SSIM;
- apply exact `torch.rot90` at 0°, 90°, 180° and 270° for the
  interpolation-free quarter-turn comparison;
- compare rotated-input encoding with the representation action applied to the
  original embedding;
- decode transformed embeddings and preserve difference/error views;
- keep image roundtrip error, latent residual and decoded disagreement as
  separate quantities;
- write artifacts atomically on the primary rank with population `n`, domains,
  angle convention, checkpoint identity and hashes.

The protocol is evaluation only: no loss term, augmentation, gradient,
optimizer update, checkpoint selection or training decision depends on it.

## Artifact Boundary

- Fixed-25 results are validation evidence, not sealed test.
- Patch dispersion is descriptive, not WSI-level uncertainty.
- PCA appearance is not a performance metric.
- Exact-quarter residual, dense-orbit local smoothness, step uniformity and
  planarity answer different questions and must not be collapsed into one
  equivariance score.
- Spec 0038 owns the completed 25x360 dense orbit.
- Spec 0044 owns the accepted advisor metric/plot package.
- Spec 0048 owns final report interpretation.

## Verification

Current code must reject selector drift, identity/order mismatch, missing
artifacts, nonfinite values, wrong angle conventions and mixed checkpoints.
Focused coverage lives in:

```bash
.venv/bin/pytest -q tests/test_fixed25_equivariance_artifacts.py
git diff --check
```
