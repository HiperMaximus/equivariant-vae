# Current Repository Status

Last updated: 2026-08-08

## Fresh-session start here

Read `AGENTS.md`, `GOAL.md`, this file, `docs/specs/README.md`, and active Spec 0011
completely. The reviewed v4 relock is the current focused commit on `main`; a substantial
uncommitted partial-v3 and broader worktree remains. Preserve unrelated work: do not
reset, checkout, or blanket-restore the tree. The user explicitly permits surgical
removal or reversion of changes proved to belong only to the failed v3 approach when v4
replacement is cleaner. Inspect each affected diff first and record what was removed;
this is not authorization to discard ambiguous or unrelated changes.

No runtime is selected. Do not execute the partial v3 controller, old-v2 `p00310`, or any
Kaggle/GitHub/Overleaf/network/remote command. Remote work always requires new explicit
user authorization even though the user removed the total Kaggle GPU-time limit.

## Objective and approved v4 intent

For correct dual-T4 `drop_last=True` training, minimize

```text
floor(real_train_patch_count / global_batch)
* synchronized_mean_steady_step_wall_time
```

Fixed-batch standings, step latency alone, largest feasible batch, throughput at an
arbitrary shared batch, compile time, deterministic RNG, and incomplete results do not
select the runtime.

The user approved this corrected direction on 2026-08-08:

- There is no minimal-toggle or recipe-simplicity objective. The fastest correct recipe
  may keep unnecessary, neutral, or redundant toggles.
- After the mandatory latest-PyPI Torch upgrade, inventory every installed/source/runtime
  value with a plausible steady-step acceleration mechanism, including experimental and
  deprecated-but-executable features.
- Remove a value only with proof of alias/duplicate effect, inert executed path,
  hardware/topology inapplicability, illegal dependency/conflict, non-steady effect,
  correctness failure, or finite-domain speed-first dominance.
- Construct inclusion-maximal compatible complete recipes. Cover every admitted value,
  every compatible pair of admitted values from distinct factors, prerequisite triple,
  and sealed complex interaction. Test
  complex interactions directly; never infer performance additivity.
- Begin from maximal bundles and descend only to rescue failure or when conditional
  removal improves projected epoch time. Do not ablate merely to simplify the winner.
- Search complete configuration and batch jointly. Do not assume monotone capacity or
  certify exact Bmax for every recipe; OOM is coordinate-local absent a separate proof.
- Kaggle GPU time is not being minimized. The finite search may use as many resumable
  sessions as necessary. The obsolete 118-probe, 12-recipe, two-interaction, four-
  contender, two-session, and `<=16 h` limits are removed.

Spec 0011 v4 completed two independent xhigh clean-context review tracks. All
blocker/high findings were integrated; both final reviews report no remaining blocker or
high finding. The focused relock commit Git-tracks the six-file canonical evidence
package. V4 is `locked / implementation-ready`; no remote execution is authorized.

## Immutable v2 evidence and required reuse

The canonical repo evidence package at `docs/data/spec0011_runtime_recipe_v2/` contains
four artifacts, preserved producer, manifest, and 309 contiguous
unique rows `p00001..p00309`, stable experiment ID
`34bf23c5370815c37a2d50cf702609f4e84fcfc99dd16e476f49096f97c67a35`, and matrix
SHA-256 `dee362d15a8c4a324d9c1f10bdec0b9aa8e1fafbf522e90d638fafcbc1571536`.
They consumed 18.19 subprocess-hours, including 6.35 hours of irregular all-batch timing.
The six files are included in the focused local relock commit; agent preflight enforces
their fresh-clone presence and tracking.
The incomplete v2 checkpoint and `p00310` never transfer or execute.

The evidence must materially reduce new work:

- verify executed measurement semantics and normalize the schema-label-only producer
  difference rather than rejecting all rows for metadata;
- accept an old row selectably only for an exact runtime/model/step/recipe/protocol/
  effective-setting/hardware identity and exact observation role;
- after the mandatory Torch upgrade, mismatching rows remain immutable priors for recipe
  order, batch wells/neighbors, VRAM/capacity expectations, and uncertainty;
- never repeat old singleton all-batch sweeps for eager, no-optimization, lite,
  channels-last, max-autotune, Python reducer, or C++ DDP;
- test old toggles again only inside genuinely new complete maximal/direct-interaction
  recipes, plus minimal labeled current-runtime reference coordinates;
- emit a verifier-checked `evidence_reuse_report` showing every accepted/prior/rejected
  row and every fresh action avoided.

No old row proves a new bundle, child, or fresh final confidence block. Old results found
irregular batch curves and non-monotone capacity, so they are valuable coordinate priors
but do not authorize monotone Bmax pruning.

## Local implementation state

A substantial partial v3 implementation exists in:

- `src/eqvae/benchmarking/runtime_recipe_search.py`
- `src/eqvae/benchmarking/runtime_recipe_bakeoff.py`
- `kaggle/kernels/runtime_recipe_bakeoff/`
- `scripts/build_kaggle_embedded_kernel.py`
- `scripts/kaggle_kernel.sh`
- the corresponding runtime-recipe tests.

It contains useful pure-policy, immutable-package, importer, checkpoint, verifier, and
packaging work, but its v3 search/budget contract is superseded and fail-closed. Preserve
sound helpers; after v4 relock, refactor them or surgically remove verified v3-only code
and tests. Do not use bulk Git restoration or run the partial controller.

Previous focused checks reached 65 passing tests plus touched Ruff/format, Bash syntax,
pure-policy BasedPyright, and `git diff --check`. Full controller typing had unresolved
diagnostics, and neither the bakeoff preflight nor full repo quality gate was run after
the contract contradiction.

## Exact next action

1. Refactor the v3 partial implementation to the accepted v4 schema and policy in the
   implementation order listed by Spec 0011.
2. Run focused mutation tests, Ruff, BasedPyright, Bash syntax, embedded-kernel preflight,
   `git diff --check`, and `./scripts/python_quality.sh`; then run two post-implementation
   adversarial reviews and update the handoff.
3. Stop locally. A Kaggle push/run/output action needs a separate explicit approval.

Only a complete independently verified artifact may emit compute finalists. Loader
starvation, paired real-data quality/LR work, selected-runtime promotion, full training,
and the continuous-`SO(2)` repeat remain later gates.
