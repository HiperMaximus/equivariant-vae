# Spec 0002: Strict Python Quality Gate

Status: active gate installed; production scope excludes historical `reference/nn`
Implementation readiness: active and passing for production Python work
Owner/workstream: agentic Python quality and local CPU verification
Latest broad verification: 687 passed, 1 expected GPU-only skip, Ruff clean, and
BasedPyright 0 errors on 2026-08-11.
Last updated: 2026-08-11

## Purpose

Make strict Python quality checks part of the agentic workflow, including the
semantic quality of the tests themselves. A green suite is useful only when each
test protects a current, decision-relevant invariant.

## Non-Goals

- Do not weaken production Python quality to accommodate historical
  exploratory code.
- Do not add global Ruff ignores.
- Do not add global BasedPyright suppressions.
- Do not import historical `reference/nn` from production `src/eqvae` code.
- Do not solve PyTorch typing limitations until the comparable VAE package
  structure is implemented.
- Do not add a prose heuristic that rewards longer or keyword-stuffed test
  docstrings. Test intent requires review against the producer and consumer.

## Test Intent Contract

A test earns its maintenance and runtime cost only when all of these are true:

1. Its docstring says why the invariant matters to a real run, artifact, safety
   boundary, or research decision. Restating the test name or assertion is not intent.
2. When it pins a non-obvious literal, fixture, tolerance, or external artifact, the
   docstring identifies whether that value is deliberate policy, a measured
   cross-check, or a derived relationship. Prefer derived relationships.
3. During review, a reviewer can name a plausible source mutation it catches; the
   docstring need not repeat an obvious mutation merely to satisfy a template.
   Presence-only tests do not claim ordering, application, causation, or execution.
4. No cheaper or stronger existing test already proves the same failure. Keep the
   smallest test that reaches the relevant seam; remove weaker duplicates.
5. The invariant is current. When production behavior is retired, delete its tests in
   the same change unless a live artifact consumer still requires compatibility.

Parameterized cases that target distinct branches are not redundant merely because
their bodies match. Conversely, separate names and docstrings do not justify two tests
that fail on the same mutation. Source-text checks are reserved for shell/bootstrap
ordering that cannot be imported and exercised directly; behavioral execution is
preferred everywhere else.

Ruff enforces docstring presence, not semantic value. This contract is therefore a
review gate for every new or changed test and a pruning rule during each touched-area
audit. Do not create a second meta-test whose only purpose is to make docstrings longer.

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
- Runtime dependencies currently include `torch` and `numpy`. The historical
  `pytorch-msssim` direct dependency was removed from `pyproject.toml` and
  `uv.lock` on 2026-06-12 because spec 0001 requires repo-owned Torch SSIM.
  User-retained exploratory `reference/nn/layers.py` still contains a reference-only
  `pytorch_msssim` import; production `src/eqvae` code must not import
  `nn` or `pytorch_msssim`.
- Developer tools:
  - Ruff for formatting and linting;
  - BasedPyright for strict static typing;
  - pytest for tests once test structure exists.

## Ruff Contract

- `select = ["ALL"]`.
- No global `ignore`.
- `extend-exclude = ["reference"]` because `reference/nn` is retained as historical
  reference material, not production Python.
- All Ruff autofixes are allowed.
- Tests may ignore only `S101` for bare `assert`.
- New global ignores require an explicit spec update and adversarial review.

Agents should run:

```bash
./scripts/python_quality.sh
```

after Python changes. The script uses the existing repo-local `.venv` without
running `uv sync`, formats first, runs `ruff check --fix`, verifies lint, runs
pytest with `PYTHONPATH=src` when Python tests exist, and then runs
BasedPyright.

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
- BasedPyright production `include` / `strict` scopes are `src/eqvae` and
  `tests`; `reference/nn` is excluded as historical reference material.

If PyTorch or `torch.nn.Module` typing blocks strict checks, do not loosen the
global config. Instead, write a small spec for the typed adapter/wrapper strategy
and prefer a local base module or local protocol that makes `__call__`,
`forward`, and initialization types line up.

Existing exploratory code already contains a `BaseModule` idea; future code
should evaluate whether current PyTorch typing still needs that pattern before
copying it into the new `src/eqvae` package.

## Production Boundary

Production Python for this gate is:

- `src/eqvae`;
- Python tests under `tests`;
- Python scripts or helpers that a later spec explicitly adds to the active
  production scope.

Historical/reference Python currently excluded from production checks:

- `reference/nn`.

`reference/nn` is intentionally left on disk for reference while the comparable VAE
implementation is built. It is not a supported import target, not part of the
quality gate, and not a source of runtime truth. Any useful idea from `reference/nn`
must be ported into typed, lint-clean `src/eqvae` code with tests before active
benchmark, training, or paper-claim code may depend on it.

## Current Gate Status

The local uv environment is Python 3.12 with CPU-only PyTorch. Strict Ruff and
BasedPyright cover `src/eqvae` and `tests`; `reference/nn` remains excluded and is
forbidden as a production import source. `CURRENT.md` owns exact gate results and
the active test-audit state so this spec does not accumulate stale test counts.

Current policy:

- new Python work must run `./scripts/python_quality.sh`;
- keep Ruff autofix enabled and accept mechanical fixes inside production
  Python;
- do not add new lint/type debt inside `src/eqvae`, tests, or active Python
  scripts;
- do not add global ignores or broad suppressions;
- document any production-scope failure in `CURRENT.md` before handing work
  back.

Benchmark-unblock route:

- implement the new comparable VAE work only under `src/eqvae`, with tests under
  `tests`;
- extract any still-useful behavior from exploratory `reference/nn` into
  typed `src/eqvae` modules instead of importing `nn`;
- empty `main.py` has been removed;
- `reference/nn` may remain as excluded historical reference material until the user
  later chooses to port, delete, or convert it to non-importable documentation;
- keep `ruff format .` and `ruff check --fix .` in `scripts/python_quality.sh`.
  Do not add Ruff global ignores for historical code;
- keep the removed `pytorch-msssim` direct dependency out of `pyproject.toml`
  and `uv.lock`; do not re-add it for historical `reference/nn`;
- `./scripts/python_quality.sh` must pass for the production Python scope before
  benchmark CLIs are considered implementation-ready.

## Acceptance Criteria

This spec is complete when:

1. `pyproject.toml` declares Python 3.12 and strict Ruff/BasedPyright settings
   for the production scope;
2. `.python-version` pins `3.12`;
3. `scripts/python_quality.sh` runs format, autofix, lint, pytest with
   `PYTHONPATH=src` when tests exist, CPU-environment assertions, and type
   checks without dependency sync;
4. agent instructions require the quality script after Python changes;
5. no global lint/type ignores are introduced;
6. any PyTorch typing workaround is specified before implementation;
7. new or changed tests satisfy the Test Intent Contract, and touched-area audits
   delete weaker duplicates or retired invariants instead of documenting them into
   apparent legitimacy.

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
