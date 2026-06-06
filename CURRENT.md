# Current Repository Status

Last updated: 2026-06-05

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE, preferably using `escnn`.

The current task is finishing planning and harness hardening before deeper code
refactors. A clean-context adversarial subagent review was run on 2026-06-05;
the converged fixes were stricter Overleaf remote validation, explicit
pull/push confirmation, token-safe remote output, stronger preflight checks,
and clearer evaluation milestone requirements.

Spec-driven development is now an active repo workflow. The first active spec is
`docs/specs/0001-translatable-normal-vae-baseline.md`.
Strict Python quality is also an active workflow via
`docs/specs/0002-strict-python-quality-gate.md`.

## Settled Decisions

- The active symmetry target is continuous `SO(2)`.
- The comparable baseline must be a normal VAE, not the previous FSQ
  autoencoder.
- The paper source of record lives in `paper/sipaim2026`.
- The tracked advisor-facing PDF is `paper/sipaim2026/sipaim2026.pdf`.
- Overleaf sync must use the safe subtree workflow.
- GitHub issue images are requirements evidence and must be inspected before
  translating issue requests into deliverables.
- Adversarial clean-context subagent reviews should be used before substantial
  workflow, architecture, evaluation, or paper-claim changes when tooling is
  available.

Decision notes live in `docs/decisions/`.
The review process lives in `docs/agentic_review_workflow.md`.

## No Longer Active

- Old conference-deadline planning is not part of the current route.
- Discrete rotation-group implementation work is not part of the current route.
- The thesis repo is not the active editing target for this phase.

## Next Concrete Steps

1. Turn the transition plan into repo code structure: configs, model factories,
   data/eval modules, and launchers.
2. Lock and implement `docs/specs/0001-translatable-normal-vae-baseline.md`.
3. Lock the Python 3.12 + Ruff + BasedPyright quality gate in
   `docs/specs/0002-strict-python-quality-gate.md`.
4. Implement the shared evaluation harness for metrics, boxplots, fixed
   25-patch artifacts, rotated-input artifacts, and latent visualizations.
5. Add targeted equivalence/equivariance tests for operations before full
   continuous `SO(2)` training runs.
6. Only then implement the steerable model path and run matched experiments.

## Current Blockers

No planning blocker is active after the adversarial workflow review. The next
blocking choices are implementation details that should be locked before full
runs:

- final input size and split policy;
- latent field/statistics policy for the first steerable VAE;
- normalization and nonlinearities to test before full experiments.

## Update Rule

Update this file after meaningful shifts in active work, blockers, or next
steps. Delete or replace stale information instead of appending contradictory
history.

## VS Code Tasks

When opening this repo in VS Code, the local workflow tasks are:

- `Agent: preflight`
- `Paper: compile SIPAIM PDF`
- `Paper: Overleaf local check`
- `Python: quality`
