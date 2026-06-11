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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened | Reopened on 2026-06-11 after correcting the latent target to `32x32x16`, replacing MSE with `L1 + 0.1 * (1 - SSIM) + beta * KL`, choosing stain-aware corruption with a corrected Tellez-style implementation, removing the final `tanh` in favor of a zero-initialized raw RGB output head, clarifying the full-mixing scalar Conv2d baseline, keeping the broad ResNet18-like residual macro-architecture, locking the branch-local ResNet-D/BlurPool-style stage transition to a fixed 5x5 separable binomial low-pass + decimation operator, recording the analytic Conv2d baseline count target, verifying Kaggle T4 metadata as `machine_shape = "NvidiaTeslaT4"`, and tightening pre-full-training benchmark requirements for single-visible-T4/dual-T4-DDP, AMP/compile, precision policies `amp_off_fp32` / `amp_conservative` / `amp_scalar_gate_relaxed`, corruption strategies `branchless_all` / `indexed_masked`, dataloader throughput, paired numerical checks, selected-runtime debug, tiny-overfit, and learned-gate health telemetry. It is not safe for broad coding yet: implementation `model_count.json` verification, future `SO(2)` count ceiling, Kaggle metadata validation/runtime two-T4 proof, quality-debt route, package/import policy, fixed validation/tiny-overfit selector generation, CPU smoke policy, and final adversarial spec review remain blockers. Final paper claims remain blocked until the sealed masked-WSI test shard is generated and locked. | Finish the spec-relock/scaffolding decision pass, then relock before implementing `src/eqvae`, `configs/spec0001`, tests, and CLI commands. After implementation/local verification, run the short Kaggle benchmark, selected-runtime debug, and tiny-overfit gates before the first 10-epoch baseline run. |
| `0002-strict-python-quality-gate.md` | active gate installed, not fully green on historical code | 146 strict Ruff errors remain after autofix; BasedPyright reports 51 strict errors; debt is in `main.py` / exploratory `src/nn`; local `.venv` must exist before no-sync checks. The benchmark-unblock route is now explicit: extract useful behavior into `src/eqvae`, remove/quarantine historical Python as non-importable documentation, remove historical `pytorch-msssim`, refresh `uv.lock`, and keep strict Ruff/BasedPyright intact. | Use that route during spec 0001 implementation, then require `./scripts/python_quality.sh` to pass before benchmark CLIs are implementation-ready. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold, not Kaggle-push-ready | Placeholder kernel intentionally refuses push; real spec 0001 launcher is not implemented yet. | Replace the placeholder with the real script launcher during spec 0001 implementation. |

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
