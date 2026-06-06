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
| `0001-translatable-normal-vae-baseline.md` | draft active, not implementation-ready | Missing Kaggle behavior inventory; open input-size, split, normalization, and nonlinearity decisions; placeholder verification commands. | Write `docs/behavior_inventory_kaggle.md`, then lock exact smoke/evaluator/artifact commands before coding. |
| `0002-strict-python-quality-gate.md` | active gate installed, not fully green on historical code | 146 strict Ruff errors remain after autofix; BasedPyright reports 51 strict errors; debt is in `main.py` / exploratory `src/nn`; local `.venv` must exist before no-sync checks. | Keep `pyproject.toml` and `uv.lock` canonical, run autofix/lint/type checks, and solve debt via `src/eqvae` cleanup or a typed-adapter spec. |
| `0003-kaggle-cli-execution-workflow.md` | draft active workflow scaffold, not Kaggle-push-ready | Kaggle CLI not installed/authenticated locally; dataset slugs unconfirmed; placeholder kernel intentionally refuses push. | Confirm dataset slugs/auth after behavior inventory, then replace the placeholder with the real script launcher. |

Keep specs current. If implementation changes the contract, update the spec in
the same workstream.
