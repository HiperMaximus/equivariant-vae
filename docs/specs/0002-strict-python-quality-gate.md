# Spec 0002: Strict Python Quality Gate

Status: draft active spec
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

after Python changes. The script formats first, then runs `ruff check --fix`,
then verifies lint and types.

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

- `uv sync --python 3.12 --group dev` creates a repo-local `.venv`.
- Linux resolves `torch==2.12.0+cpu`.
- `torch.cuda.is_available()` is `False`, as intended for local laptop tests.
- Ruff format and `ruff check --fix` run successfully and leave Ruff green.
- BasedPyright runs successfully but reports 51 strict typing errors in the
  historical exploratory files `src/nn/layers.py` and `src/nn/resnet18.py`.

The BasedPyright failures are mostly `reportAny` from PyTorch module call paths,
plus one redeclaration. This is the expected next blocker to solve during the
new `src/eqvae` implementation or a dedicated typed-PyTorch adapter spec. Do
not make the gate green by weakening global strictness.

## Acceptance Criteria

This spec is complete when:

1. `pyproject.toml` declares Python 3.12 and strict Ruff/BasedPyright settings;
2. `.python-version` pins `3.12`;
3. `scripts/python_quality.sh` runs format, autofix, lint, and type checks;
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
uv sync --python 3.12 --group dev
./scripts/python_quality.sh
```

If `uv` needs to install Python or dependencies, agents must ask for permission
before running the full gate in a restricted environment.

## Related Files

- `pyproject.toml`
- `.python-version`
- `scripts/python_quality.sh`
- `AGENTS.md`
- `docs/spec_driven_development.md`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
