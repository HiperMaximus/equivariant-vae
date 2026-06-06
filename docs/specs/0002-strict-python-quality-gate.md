# Spec 0002: Strict Python Quality Gate

Status: active gate installed; not fully green on historical code
Implementation readiness: active for new Python work
Owner/workstream: agentic Python quality and local CPU verification
Last updated: 2026-06-05

## Purpose

Make strict Python quality checks part of the agentic workflow so future agents
fix trivial formatting, linting, and typing issues automatically instead of
handing them back to the user.

## Non-Goals

- Do not weaken the project to accommodate historical exploratory code.
- Do not add global Ruff ignores.
- Do not add global BasedPyright suppressions.
- Do not solve PyTorch typing limitations until the comparable VAE package
  structure is implemented.

## Environment Contract

- Python target: 3.12, matching the intended Kaggle-style runtime.
- Environment manager: `uv`.
- Project marker: `.python-version` with `3.12`.
- Lockfile: `uv.lock` should be tracked.
- Source of truth:
  - `pyproject.toml` declares direct dependencies and tool configuration;
  - `uv.lock` captures the resolved local environment;
  - root `requirements.txt` is not allowed.
- Local laptop tests use CPU-only PyTorch. GPU training belongs to Kaggle.
- Linux PyTorch resolves from the PyTorch CPU wheel index through
  `tool.uv.sources`.
- Runtime dependencies currently include `torch`, `pytorch-msssim`, and `numpy`.
- Developer tools:
  - Ruff for formatting and linting;
  - BasedPyright for strict static typing;
  - pytest for tests once test structure exists.

## Ruff Contract

- `select = ["ALL"]`.
- No global `ignore`.
- All Ruff autofixes are allowed.
- Tests may ignore only `S101` for bare `assert`.
- New global ignores require an explicit spec update and adversarial review.

Agents should run:

```bash
./scripts/python_quality.sh
```

after Python changes. The script uses the existing repo-local `.venv` without
running `uv sync`, formats first, runs `ruff check --fix`, verifies lint, runs
pytest when Python tests exist, and then runs BasedPyright.

If `.venv` is missing or stale, ask the user before running:

```bash
uv sync --locked --python 3.12 --group dev
```

Do not use `ruff.toml`; it shadows the strict settings in `pyproject.toml`.

## BasedPyright Contract

- `typeCheckingMode = "strict"`.
- No global type suppressions.
- Missing imports and missing type stubs are errors.
- `reportAny` and ignore comments without explicit rules are errors.

If PyTorch or `torch.nn.Module` typing blocks strict checks, do not loosen the
global config. Instead, write a small spec for the typed adapter/wrapper strategy
and prefer a local base module or local protocol that makes `__call__`,
`forward`, and initialization types line up.

Existing exploratory code already contains a `BaseModule` idea; future code
should evaluate whether current PyTorch typing still needs that pattern before
copying it into the new `src/eqvae` package.

## Current Gate Status

As of 2026-06-05:

- `uv sync --locked --python 3.12 --group dev` creates a repo-local `.venv`.
- Linux resolves `torch==2.12.0+cpu`.
- `torch.cuda.is_available()` is `False`, as intended for local laptop tests.
- Strict Ruff settings live in `pyproject.toml`. A stale `ruff.toml` previously
  shadowed them and has been removed.
- The no-sync quality gate now reaches the strict Ruff config. Ruff autofixed 14
  formatting issues in historical `src/nn/layers.py`, then reported 146
  remaining errors in `main.py` and historical exploratory `src/nn` files.
- BasedPyright runs successfully but reports 51 strict typing errors in the
  historical exploratory files `src/nn/layers.py` and `src/nn/resnet18.py`.

Strict Ruff and BasedPyright failures in historical exploratory code are the
expected next blocker to solve during the new `src/eqvae` implementation, a
historical-code cleanup, or a dedicated typed-PyTorch adapter spec. Do not make
the gate green by weakening global strictness.

Interim policy until the historical debt is resolved:

- new Python work must run `./scripts/python_quality.sh`;
- keep Ruff autofix enabled and accept mechanical fixes;
- do not add new lint/type debt outside already documented historical files;
- do not add global ignores or broad suppressions;
- document any remaining failure count in `CURRENT.md` before handing work back.

## Acceptance Criteria

This spec is complete when:

1. `pyproject.toml` declares Python 3.12 and strict Ruff/BasedPyright settings;
2. `.python-version` pins `3.12`;
3. `scripts/python_quality.sh` runs format, autofix, lint, pytest when tests
   exist, CPU-environment assertions, and type checks without dependency sync;
4. agent instructions require the quality script after Python changes;
5. no global lint/type ignores are introduced;
6. any PyTorch typing workaround is specified before implementation.

## Verification Commands

Local, no-network checks:

```bash
python3 -c 'import pathlib, tomllib; tomllib.loads(pathlib.Path("pyproject.toml").read_text())'
bash -n scripts/python_quality.sh
```

Full quality gate, may need dependency sync:

```bash
uv sync --locked --python 3.12 --group dev
./scripts/python_quality.sh
```

If `uv` needs to install Python or dependencies, agents must ask for permission
before running the full gate in a restricted environment.

## Related Files

- `pyproject.toml`
- `.python-version`
- `uv.lock`
- `.gitignore`
- `scripts/python_quality.sh`
- `scripts/agent_preflight.sh`
- `AGENTS.md`
- `docs/spec_driven_development.md`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
