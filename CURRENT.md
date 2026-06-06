# Current Repository Status

Last updated: 2026-06-05

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE, preferably using `escnn`.

The current task is finishing planning and harness hardening before deeper code
refactors. Clean-context adversarial subagent reviews were run on 2026-06-05;
the latest fixes tightened dependency truth, strict Ruff resolution, no-network
quality checks, preflight guards, spec readiness, behavior-inventory gates, and
handoff-memory requirements. A local Kaggle CLI execution scaffold now exists,
but it is not Kaggle-push-ready.

Spec-driven development is now an active repo workflow. The first active spec is
`docs/specs/0001-translatable-normal-vae-baseline.md`, but it is still a draft
and is not implementation-ready.
Strict Python quality is also an active workflow via
`docs/specs/0002-strict-python-quality-gate.md`.
Kaggle CLI execution is scaffolded via
`docs/specs/0003-kaggle-cli-execution-workflow.md`,
`docs/kaggle_cli_workflow.md`, `scripts/kaggle_kernel.sh`, and
`kaggle/kernels/non_eq_vae_debug`.
The local uv environment is CPU-only for PyTorch. Strict Ruff settings are
canonical in `pyproject.toml`; do not add `ruff.toml`. The no-sync quality gate
verified Python 3.12, `torch==2.12.0+cpu`, and CUDA unavailable. Strict Ruff
autofixed 14 historical formatting issues and then reported 146 remaining
errors, all in `main.py` / historical exploratory `src/nn` files. A direct
BasedPyright run reports 51 strict errors in historical exploratory `src/nn`
files. Solve this in the new `src/eqvae` implementation, a historical-code
cleanup, or a dedicated typed-PyTorch adapter spec rather than weakening global
strictness.

Immediate next action: write the Kaggle behavior inventory in
`docs/behavior_inventory_kaggle.md` from `kaggle/train_runs` and
`kaggle/dataset_generation`. Do not implement spec 0001 until that inventory
exists and `docs/specs/0001-translatable-normal-vae-baseline.md` is locked as
implementation-ready.

Kaggle-specific handoff: `scripts/kaggle_kernel.sh validate` works locally, but
`kaggle` is not installed/authenticated on this laptop yet. Do not run
`KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push` until the behavior
inventory exists, spec 0001 is locked, dataset slugs are confirmed, the
placeholder guard is removed from `kaggle/kernels/non_eq_vae_debug/run.py`, and
the user explicitly approves the remote write.

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

1. Write `docs/behavior_inventory_kaggle.md` with the current data, training,
   resume, evaluation, and artifact behavior from the Kaggle notebooks/scripts.
2. Lock `docs/specs/0001-translatable-normal-vae-baseline.md` as
   implementation-ready with exact smoke/evaluator/artifact commands.
3. Confirm Kaggle dataset slugs and authentication plan for the CLI-managed
   script-kernel workflow.
4. Resolve or explicitly baseline the strict Ruff/BasedPyright historical debt
   without weakening global quality settings.
5. Turn the transition plan into repo code structure: configs, model factories,
   data/eval modules, and launchers.
6. Lock the Python 3.12 + Ruff + BasedPyright quality gate in
   `docs/specs/0002-strict-python-quality-gate.md`.
7. Implement the shared evaluation harness for metrics, boxplots, fixed
   25-patch artifacts, rotated-input artifacts, and latent visualizations.
8. Add targeted equivalence/equivariance tests for operations before full
   continuous `SO(2)` training runs.
9. Only then implement the steerable model path and run matched experiments.

## Current Blockers

- `docs/specs/0001-translatable-normal-vae-baseline.md` is draft active, not
  implementation-ready.
- The Kaggle behavior inventory has not yet been written.
- Exact smoke-test, evaluator, and artifact-generation commands are still
  placeholders in spec 0001.
- Kaggle CLI is not installed/authenticated locally, and real dataset slugs are
  not confirmed in `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json`.
- Strict Python quality is intentionally not fully green on historical
  exploratory code: 146 Ruff errors remain after autofix, and BasedPyright
  reports 51 strict errors. New work must not add debt or weaken the gate.
- The next blocking choices to lock before full runs are:

- final input size and split policy;
- latent field/statistics policy for the first steerable VAE;
- normalization and nonlinearities to test before full experiments.

## Update Rule

Update this file after meaningful shifts in active work, blockers, or next
steps, and before handing work back from a partial state. Each handoff update
should make clear:

- what changed;
- what is currently in progress;
- exactly where the agent left off;
- the next concrete action;
- active blockers or decisions needed;
- verification run and remaining failures.

Delete or replace stale information instead of appending contradictory history.

## VS Code Tasks

When opening this repo in VS Code, the local workflow tasks are:

- `Agent: preflight`
- `Paper: compile SIPAIM PDF`
- `Paper: Overleaf local check`
- `Python: quality`
