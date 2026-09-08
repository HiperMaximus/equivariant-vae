# Open Follow-Ups

Only unresolved, currently actionable debt belongs here. Delete an entry when
it is resolved; promote work requiring a contract into a spec.

## Repository Tooling

- `uv build` can emit an empty wheel when any symlink to `src/eqvae` exists
  elsewhere in the tree because the sdist walk deduplicates by real path. This
  does not affect editable installs or Kaggle execution. Before publishing a
  wheel, reproduce and fix the package walk.
- `scripts/build_kaggle_embedded_kernel.py` embeds ZIP member mtimes, so two
  builds from identical source can produce different `run.py` byte hashes.
  Make member order and `ZipInfo.date_time` deterministic before claiming
  byte-reproducible wrappers.
- Debug `step.py` and the full runner's `_run_train_step` are separate
  implementations. Consolidate them or add a parity check before changing
  shared training semantics.
- Training CSV loss/gradient telemetry is rank-local. Keep that limitation in
  plots unless a global reduction is implemented.

## Report And Paper

- The report builder uses broad file-level type suppressions. Narrow them if
  the builder receives substantive Python changes.
- `paper/sipaim2026` does not yet contain the accepted final experiment results
  and remains outside the completed professor-report scope. A paper task must
  replace placeholders and stale claims, compile the tracked PDF and use the
  guarded Overleaf workflow.
- A WSI patch-attribution map requires new instrumented frozen-checkpoint
  inference; current accepted outputs do not contain patch-level gates.

## Environment

- `/tmp` is a quota-limited tmpfs and can create false test failures. Put agent
  scratch in `.agent_tmp/` and leave `TMPDIR` unset for
  `./scripts/python_quality.sh`, which redirects heavy pytest temp data to
  `runs/local_tmp`.
