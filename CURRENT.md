# Current Repository Status

Last updated: 2026-06-10

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
The Kaggle behavior inventory now lives at
`docs/behavior_inventory_kaggle.md`. Dataset slugs were confirmed through the
Kaggle CLI, and the debug kernel metadata now points at
`maximusshtefan/patches-pre-shuffled-ubc-ocean`.
Important dataset nuance: that dataset is the confirmed pre-shuffled
train/validation patch source, with `ubc_train_shuffled.*` and
`ubc_ocean_valid.*` files verified through the Kaggle CLI on 2026-06-10. It does
not contain a held-out test shard. The
`kaggle/generate_dataset_Classification_With_Masks` notebook is the current
test-set-generation starting point, but as committed it still writes train/valid
splits rather than `test` files. User-confirmed split intent: train/validation
uses WSIs without supplemental masks; WSIs with non-exhaustive supplemental masks
are reserved for the held-out autoencoder test set and later supervised
experiments.
The local uv environment is CPU-only for PyTorch. Strict Ruff settings are
canonical in `pyproject.toml`; do not add `ruff.toml`. The no-sync quality gate
verified Python 3.12, `torch==2.12.0+cpu`, and CUDA unavailable. Strict Ruff
autofixed 14 historical formatting issues and then reported 146 remaining
errors, all in `main.py` / historical exploratory `src/nn` files. A direct
BasedPyright run reports 51 strict errors in historical exploratory `src/nn`
files. Solve this in the new `src/eqvae` implementation, a historical-code
cleanup, or a dedicated typed-PyTorch adapter spec rather than weakening global
strictness.

Immediate next action: use `docs/behavior_inventory_kaggle.md` to lock
`docs/specs/0001-translatable-normal-vae-baseline.md` as
implementation-ready, especially exact smoke-test, evaluator, artifact, and
debug Kaggle launcher commands, plus the sealed held-out masked-WSI test-set plan
for final evaluation.

Kaggle-specific handoff: `scripts/kaggle_kernel.sh validate` and
`scripts/kaggle_kernel.sh check` worked locally on 2026-06-06 with Kaggle CLI
2.2.1, but Kaggle authentication is a user-local secret and must be treated as
permission-gated. Do not run
`KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push` until spec 0001 is
locked, the placeholder guard is removed from
`kaggle/kernels/non_eq_vae_debug/run.py`, and the user explicitly approves the
remote write.

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

1. Lock `docs/specs/0001-translatable-normal-vae-baseline.md` as
   implementation-ready with exact smoke/evaluator/artifact commands.
2. Replace the placeholder Kaggle debug kernel with a real launcher only after
   spec 0001 is locked.
3. Resolve or explicitly baseline the strict Ruff/BasedPyright historical debt
   without weakening global quality settings.
4. Turn the transition plan into repo code structure: configs, model factories,
   data/eval modules, and launchers.
5. Lock the Python 3.12 + Ruff + BasedPyright quality gate in
   `docs/specs/0002-strict-python-quality-gate.md`.
6. Implement the shared evaluation harness for metrics, boxplots, fixed
   25-patch artifacts, rotated-input artifacts, and latent visualizations.
7. Add targeted equivalence/equivariance tests for operations before full
   continuous `SO(2)` training runs.
8. Only then implement the steerable model path and run matched experiments.

## Current Blockers

- `docs/specs/0001-translatable-normal-vae-baseline.md` is draft active, not
  implementation-ready.
- Exact smoke-test, evaluator, and artifact-generation commands are still
  placeholders in spec 0001.
- The exact held-out masked-WSI test shard source must be generated, uploaded,
  and locked before final paper claims. Train/validation are available in the
  confirmed pre-shuffled patch dataset. Supplemental masks are non-exhaustive, so
  test generation and later supervised experiments must not treat unmasked
  regions as exhaustive negative labels.
- The Kaggle debug kernel still has a `NOT_IMPLEMENTATION_READY` placeholder and
  must not be pushed.
- Strict Python quality is intentionally not fully green on historical
  exploratory code: 146 Ruff errors remain after autofix, and BasedPyright
  reports 51 strict errors. New work must not add debt or weaken the gate.
- The next blocking choices to lock before full runs are:

- final input size and split/masked-test-source policy;
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
