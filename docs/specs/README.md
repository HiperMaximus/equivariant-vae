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
| `0001-translatable-normal-vae-baseline.md` | draft active / reopened | Reopened on 2026-06-11 after correcting the latent target to `32x32x16`, replacing MSE with `L1 + 0.1 * (1 - SSIM) + beta * KL`, choosing stain-aware corruption with a corrected Tellez-style implementation, clarifying the full-mixing scalar Conv2d baseline, and adding short Kaggle runtime benchmark requirements for single/dual T4, AMP off/on, and compile off/on. It is not safe for broad coding yet: parameter/FLOP count, quality-debt route, package/import policy, config parser/dependency policy, fixed-25 selector generation, CPU smoke policy, and final adversarial spec review remain blockers. Final paper claims remain blocked until the sealed masked-WSI test shard is generated and locked. | Finish the spec-relock/scaffolding decision pass, then relock before implementing `src/eqvae`, `configs/spec0001`, tests, and CLI commands. After implementation/local verification, run the short Kaggle benchmark before the first 10-epoch baseline run. |
| `0002-strict-python-quality-gate.md` | active gate installed, not fully green on historical code | 146 strict Ruff errors remain after autofix; BasedPyright reports 51 strict errors; debt is in `main.py` / exploratory `src/nn`; local `.venv` must exist before no-sync checks. | Keep `pyproject.toml` and `uv.lock` canonical, run autofix/lint/type checks, and solve debt via `src/eqvae` cleanup or a typed-adapter spec. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold, not Kaggle-push-ready | Placeholder kernel intentionally refuses push; real spec 0001 launcher is not implemented yet. | Replace the placeholder with the real script launcher during spec 0001 implementation. |

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
