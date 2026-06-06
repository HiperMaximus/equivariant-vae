# Repo Goal

This repository supports the SIPAIM 2026 paper and the experiments behind it.
The thesis repository is separate and will be updated later from stable paper
results.

## North Star

Compare two genuinely comparable histopathology patch representation learners:

1. A non-equivariant denoising VAE built only from operations that have a clear
   path to the steerable implementation.
2. A continuous `SO(2)`-steerable denoising VAE, preferably using `escnn`, with
   matched data, schedule, latent target, training budget, and evaluation.

The comparison should answer whether continuous rotation-equivariant structure
improves reconstruction, latent behavior, robustness, or downstream usefulness
for histopathology patches.

## Active Source Of Truth

- Hard repo instructions:
  `AGENTS.md`
- Architecture transition plan:
  `docs/equivariant_vae_transition_plan.md`
- Current handoff/status:
  `CURRENT.md`
- Issue-derived requirements and deliverables:
  `docs/repo_goal_and_requirements.md`
- Issue image inventory:
  `docs/issue_image_inventory.md`
- Settled decisions:
  `docs/decisions/README.md`
- Agentic adversarial review workflow:
  `docs/agentic_review_workflow.md`
- Spec-driven development workflow:
  `docs/spec_driven_development.md` and `docs/specs/`
- Strict Python quality workflow:
  `docs/specs/0002-strict-python-quality-gate.md`
- SIPAIM paper source:
  `paper/sipaim2026`
- Overleaf sync workflow:
  `docs/overleaf_sync_workflow.md`

## Current Paper Target

- Conference: SIPAIM 2026.
- Format: IEEE conference full paper.
- Local paper source: `paper/sipaim2026`.
- Overleaf project:
  https://www.overleaf.com/project/69c614433cbc9e46cf226d24
- Tracked advisor-facing PDF: `paper/sipaim2026/sipaim2026.pdf`.

Before pushing paper changes to Overleaf or GitHub, refresh the PDF with:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

## Do Not Lose These Requirements

- Report SSIM, MAE, MSE, and PSNR with mean, standard deviation, and sample
  count `n`.
- Produce boxplots for the reconstruction metrics.
- Produce a training/evaluation dashboard analogous to the issue screenshots.
- Keep a fixed 25-patch validation set for qualitative reconstructions and
  visual checks.
- Generate original/reconstructed folders or grids for those 25 fixed patches.
- Add rotated-input qualitative artifacts using fixed continuous angles.
- Add `paper/sipaim2026/figures/rotated_input_vs_latent_grid.*`, comparing
  ground truth, rotated-input reconstruction, transformed-latent reconstruction,
  and error maps for the same patch/angle set.
- Implement an EQ-VAE-style latent visualization: top principal components,
  latent maps, transformed latent maps, and error/difference maps.
- Validate equivariance of nonlinearities, normalization, upsampling, VAE
  sampling, and latent statistics before running the full experiment.
