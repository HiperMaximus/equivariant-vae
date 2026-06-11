# Current Repository Status

Last updated: 2026-06-11

## Active Workstream

Build the repo toward a fair SIPAIM 2026 comparison between:

1. a non-equivariant normal denoising VAE whose operations translate to the
   steerable implementation; and
2. a continuous `SO(2)` steerable denoising VAE, preferably using `escnn`.

The current task is relocking the translatable normal VAE baseline spec after
architecture/objective corrections. Clean-context adversarial subagent reviews
were run on 2026-06-05, 2026-06-10, and 2026-06-11. The 2026-06-11 pass
confirmed that the previous `4x4` latent target was inconsistent with the
FSQ-successor spatial-coherence goal and that the historical HED corruptor must
not be copied as-is. A local Kaggle CLI execution scaffold now exists, but it is
not Kaggle-push-ready.

Spec-driven development is now an active repo workflow. The first active spec is
`docs/specs/0001-translatable-normal-vae-baseline.md`, now reopened as
`draft active` and not implementation-ready. The reopened direction is:
`32x32x16` scalar Gaussian latent, no FSQ quantizer or learned bottleneck scale
`s`, corrected Tellez-style HED/OD stain corruption plus per-image Gaussian
noise `Uniform(0.0, 0.05)`, full-mixing scalar Conv2d baseline channels with a
shared gated scalar activation family, future `SO(2)` radial gates only for
nontrivial irrep fields, and `L1 + 0.1 * (1 - SSIM) + beta * KL`. It is not
final-paper-claim-ready until the sealed masked-WSI test shard is generated and
locked.
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
not contain a held-out test shard. The split was checked against official
UBC-OCEAN metadata on 2026-06-10: train has 322 non-TMA WSIs and 300000 patch
rows, validation has 39 non-TMA WSIs and 30000 patch rows, train/validation WSI
overlap is zero, and both splits have zero overlap with the 152 supplemental-mask
WSIs. The exact masked holdout candidate list is
`docs/data/ubc_ocean_masked_holdout_ids.csv`; the sealed test shard itself still
needs to be generated. The
`kaggle/generate_dataset_Classification_With_Masks` notebook is the current
test-set-generation starting point, but as committed it still writes train/valid
splits rather than `test` files. User-confirmed split intent: train/validation
uses WSIs without supplemental masks; WSIs with non-exhaustive supplemental masks
are reserved for the held-out autoencoder test set and later supervised
experiments.
A clean-context adversarial review pass on 2026-06-10 checked the agentic
workflow and Kaggle data contract. It found and fixed stale onboarding references
to the Kaggle mask notebook, missing preflight coverage for the masked-holdout
CSV, loose Kaggle spec-index readiness checks, and an ambiguity in the patch CSV
metadata schema. The new holdout CSV is tracked so repo preflight can verify it as
tracked.
An additional clean-context adversarial coding-readiness audit on 2026-06-11
found that the repo is not yet safe for broad spec 0001 implementation. It is
ready for a spec-relock/scaffolding decision pass only. The audit added or
confirmed blockers for parameter/FLOP counting, residual-off policy,
strict-quality debt route, package/import policy, config parser/dependency
policy, fixed-25 selector generation, CPU compile/float16 smoke constraints,
baseline rotated/latent artifact semantics, and local-vs-Kaggle acceptance
separation.
The local uv environment is CPU-only for PyTorch. Strict Ruff settings are
canonical in `pyproject.toml`; do not add `ruff.toml`. The no-sync quality gate
verified Python 3.12, `torch==2.12.0+cpu`, and CUDA unavailable. Strict Ruff
autofixed 14 historical formatting issues and then reported 146 remaining
errors, all in `main.py` / historical exploratory `src/nn` files. A direct
BasedPyright run reports 51 strict errors in historical exploratory `src/nn`
files. Solve this in the new `src/eqvae` implementation, a historical-code
cleanup, or a dedicated typed-PyTorch adapter spec rather than weakening global
strictness.

Immediate next action: finish relocking
`docs/specs/0001-translatable-normal-vae-baseline.md` before implementation.
The benchmark contract is now written, but the runtime result cannot be selected
until the benchmarkable code exists. Resolve the remaining implementation-relock
blockers in spec 0001, especially parameter/FLOP count for the reopened
`32x32x16` architecture schedule, strict-quality and config/dependency
decisions, and final clean-context adversarial spec review. Then implement
`src/eqvae`, `configs/spec0001`, tests, and CLI commands in the milestone slices
recorded in spec 0001. After local verification, run the short Kaggle runtime
benchmark to choose single/dual T4, per-device/global batch, AMP, and compile
settings before the first 10-epoch full run.

Kaggle-specific handoff: `scripts/kaggle_kernel.sh validate` and
`scripts/kaggle_kernel.sh check` worked locally on 2026-06-06 with Kaggle CLI
2.2.1, but Kaggle authentication is a user-local secret and must be treated as
permission-gated. Do not run
`KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push` until the placeholder
guard is removed from `kaggle/kernels/non_eq_vae_debug/run.py`, the real spec
0001 launcher exists, local verification passes, and the user explicitly
approves the remote write.

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

1. Finish the spec 0001 relock: parameter/FLOP count, final `32x32x16`
   channel/future-field schedule, strict-quality route, package/import policy,
   config parser/dependency policy, fixed-25 selector plan, CPU smoke policy, and
   final clean-context adversarial spec review.
2. Mark spec 0001 `locked / implementation-ready` only after those relock
   blockers are resolved.
3. Add the `src/eqvae` package skeleton and `configs/spec0001` files required by
   the relocked spec.
4. Implement synthetic patch data, patch-shard loading, corrected stain
   corruption, shared gated scalar activation policy, and the non-equivariant VAE
   factory.
5. Add the exact spec 0001 tests and CLI commands for smoke, train, resume,
   evaluate, and artifacts.
6. Replace the placeholder Kaggle debug kernel with the real launcher only after
   local spec 0001 verification passes.
7. Run the short Kaggle runtime benchmark after explicit user permission and
   record the selected single/dual T4, per-device/global batch, AMP, and compile
   config before the first 10-epoch baseline run.
8. Resolve or explicitly baseline the strict Ruff/BasedPyright historical debt
   without weakening global quality settings.
9. Lock the Python 3.12 + Ruff + BasedPyright quality gate in
   `docs/specs/0002-strict-python-quality-gate.md`.
10. Implement the shared evaluation harness for metrics, boxplots, fixed
   25-patch artifacts, rotated-input artifacts, and latent visualizations.
11. Add targeted equivalence/equivariance tests for operations before full
   continuous `SO(2)` training runs.
12. Only then implement the steerable model path and run matched experiments.

## Current Blockers

- Spec 0001 is reopened and not implementation-ready. Remaining
  implementation-relock blockers are listed in
  `docs/specs/0001-translatable-normal-vae-baseline.md` and include
  parameter/FLOP count, strict-quality route, package/import policy,
  config/dependency policy, fixed-25 selector generation, CPU smoke policy, and
  final adversarial spec review.
- The first full Kaggle run remains blocked after implementation until the short
  runtime benchmark selects single/dual T4, per-device/global batch, AMP, and
  compile settings.
- The exact held-out masked-WSI test shard must be generated, uploaded, and
  locked before final paper claims. The 152-image candidate pool is documented in
  `docs/data/ubc_ocean_masked_holdout_ids.csv`, and train/validation are
  available in the confirmed pre-shuffled patch dataset. Supplemental masks are
  non-exhaustive, so test generation and later supervised experiments must not
  treat unmasked regions as exhaustive negative labels.
- The Kaggle debug kernel still has a `NOT_IMPLEMENTATION_READY` placeholder and
  must not be pushed until the real spec 0001 launcher is implemented and
  verified.
- Strict Python quality is intentionally not fully green on historical
  exploratory code: 146 Ruff errors remain after autofix, and BasedPyright
  reports 51 strict errors. New work must not add debt or weaken the gate.
- The next blocking choices before final paper claims are the exact sealed
  masked-WSI test-shard artifact, upload slug, and mount-path verification.
  The next blocking choices before the steerable model are the latent
  field/statistics policy for nontrivial `SO(2)` latents and any normalization
  ablation.

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
