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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened; scaffold is `scaffold_schema_ready`; model count is `topology_count_ready`; local data/metrics is `data_metrics_ready`; local selector/dataloader is `fixed_selectors_dataloader_ready`; local CPU dataloader pre-test is `local_benchmark_pretest_ready`; model/loss train-step is `model_loss_train_step_ready`; local HED/stain corruption QA is `corruption_ready`; kaggle smoke is `kaggle_smoke_ready`; synthetic no-dataset setup smoke is `kaggle_setup_smoke_ready` and has one remote `smoke_pass` | Reopened on 2026-06-11 after correcting the latent target to `32x32x16`, replacing MSE with `L1 + 0.1 * (1 - SSIM) + beta * KL`, choosing stain-aware corruption, removing the final `tanh`, clarifying the Conv2d baseline, locking the fixed binomial low-pass stage transition, recording the Conv2d count target, verifying Kaggle T4 metadata as `machine_shape = "NvidiaTeslaT4"`, and tightening runtime, dataloader, paired-check, gate-health, selected-runtime debug, and tiny-overfit requirements. Narrow local slices implemented on 2026-06-12 cover topology count, data/metrics, selector/dataloader, local dataloader pre-test, and model/loss train-step contracts. The 2026-06-13 HED/stain corruption local slice now implements scikit-image-compatible PyTorch HED semantics, and the capped Kaggle smoke now verifies metadata-carrying UBC-format batches, real corruptor plumbing, local synthetic train steps with at least one applied corruption, and one clean-validation batch without becoming runtime selection evidence. The first remote real-data smoke version finished as `ERROR` with `ModuleNotFoundError: No module named 'eqvae'` because the sibling payload was not uploaded, so it produced no benchmark evidence. The 2026-06-17 setup-smoke slice now generates a single-file embedded no-dataset kernel and upload-simulation test for Kaggle packaging/API/import/artifact plumbing only; remote version 1 completed with non-promotable setup `smoke_pass`. It is not safe for broad coding yet: fixed real 25-patch visual QA, branchless/indexed runtime corruption checks, selected-runtime debug, real train/resume/evaluator/artifact writers, future `SO(2)` count ceiling, Kaggle runtime two-T4 proof, real fixed selector generation from real Kaggle shards, and final adversarial spec review remain blockers. Final paper claims remain blocked until the sealed masked-WSI test shard is generated and locked. | Do not rerun the real-data smoke until its source delivery is migrated to embedded single-file packaging or another proved mechanism. Do not treat local `schema_pass`, `local_pass`, setup `smoke_pass`, failed remote v1 output, or hardened real-data `smoke_pass` outputs as Kaggle runtime selection. |
| `0002-strict-python-quality-gate.md` | active gate installed; production scope excludes historical `src/nn`; passing | Empty `main.py` was deleted and `pytorch-msssim` was removed from `pyproject.toml`/`uv.lock` on 2026-06-12. Historical exploratory `src/nn` remains as reference material by user decision, excluded from Ruff/BasedPyright production scopes, and forbidden as an import source for `src/eqvae`. The latest `./scripts/python_quality.sh` passed for production Python with 75 tests and 0 BasedPyright errors. Local `.venv` must exist before no-sync checks. | Keep new work in typed `src/eqvae`; do not import `src.nn`; keep `./scripts/python_quality.sh` passing before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold; synthetic setup-smoke remote v1 passed | The setup-smoke kernel now launches a generated embedded no-dataset setup path and remote v1 completed with non-promotable `smoke_pass`. The real-data capped smoke still needs embedded packaging before rerun. Full benchmark/full-run launchers are still not implemented. | Use the setup-smoke result only as Kaggle packaging/API/import/artifact evidence. Remote writes still require explicit user permission and `KAGGLE_PUSH_CONFIRMED=1`. |

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
