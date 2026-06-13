# Specs

This directory contains implementation and experiment specs.

Use these specs to define what "done" means before changing code, evaluation
logic, paper artifacts, or workflow tooling.

Read first:

1. `../spec_driven_development.md`
2. `template.md`
3. the active specs in the table below

## Status Values

- `draft`: useful planning text, not ready for implementation.
- `draft active`: active workstream, but blocked by open questions or missing
  verification details.
- `locked / implementation-ready`: enough contract, acceptance criteria, and
  commands exist to start coding.
- `implemented`: accepted and verified.
- `superseded`: retained only for history, with a pointer to the replacement.

## Active Specs

| Spec | Status | Blocked By | Next Action |
| --- | --- | --- | --- |
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; scaffold is `scaffold_schema_ready`; model count is `topology_count_ready`; local data/metrics is `data_metrics_ready`; local selector/dataloader is `fixed_selectors_dataloader_ready`; local CPU dataloader pre-test is `local_benchmark_pretest_ready`; model/loss train-step is `model_loss_train_step_ready`; HED/stain corruption contract is `corruption_contract_ready` | Reopened on 2026-06-11 after correcting the latent target to `32x32x16`, replacing MSE with `L1 + 0.1 * (1 - SSIM) + beta * KL`, choosing stain-aware corruption, removing the final `tanh`, clarifying the Conv2d baseline, locking the fixed binomial low-pass stage transition, recording the Conv2d count target, verifying Kaggle T4 metadata as `machine_shape = "NvidiaTeslaT4"`, and tightening runtime, dataloader, paired-check, gate-health, selected-runtime debug, and tiny-overfit requirements. Narrow local slices implemented on 2026-06-12 cover topology count, data/metrics, selector/dataloader, local dataloader pre-test, and model/loss train-step contracts. The 2026-06-13 HED/stain corruption spec-lock pass now locks scikit-image-compatible PyTorch HED semantics, conservative/default and FSQ-wide profiles, tiny residual-axis jitter wording, semantic stateless RNG without rank in the seed, branchless-all as the first execution strategy, output/metadata contracts, and non-promotable `benchmark/stain_corruptor_qa.json` plus `benchmark/corruption_checks.csv` evidence. It is not safe for broad coding yet: corruption implementation, real train/resume/evaluator/artifact writers, future `SO(2)` count ceiling, Kaggle metadata validation/runtime two-T4 proof, real fixed selector generation from real Kaggle shards, and final adversarial spec review remain blockers. Final paper claims remain blocked until the sealed masked-WSI test shard is generated and locked. | Implement the next narrow local corruption correctness/QA slice only after explicit user approval. Do not treat local `schema_pass` or `local_pass` runtime/pre-test/QA outputs as Kaggle runtime selection. Run the short Kaggle benchmark, selected-runtime debug, and tiny-overfit gates only with explicit permission after local implementation acceptance. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Empty `main.py` was deleted and `pytorch-msssim` was removed from `pyproject.toml`/`uv.lock` on 2026-06-12. Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. The latest `./scripts/python_quality.sh` passed for production Python with 60 tests and 0 BasedPyright errors. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold, not Kaggle-push-ready | Placeholder kernel intentionally refuses push; real spec 0001 launcher is not implemented yet. | Replace the placeholder with the real script launcher during spec 0001 implementation. |

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
