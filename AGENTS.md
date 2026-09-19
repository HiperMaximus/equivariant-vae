# Repository Instructions

This is a research repository for the matched normal versus continuous-`SO(2)`
VAE experiment. It is not production software.

## Read First

Read `CURRENT.md`, `GOAL.md`, and the active scientific spec relevant to the
task. Read a specialized workflow document only when the task actually uses
that workflow.

## Working Rules

1. Make the smallest change that answers the scientific question. A parameter
   rerun changes the existing config and reuses the same thin kernel; it
   does not create a new architecture, schema, helper layer, process document,
   or broad test harness.
2. Add reusable code only when the mathematics is genuinely shared or a focused
   test is needed to distinguish correct from incorrect numerical behavior.
3. One-shot experimental runners are fail-fast: use only exact
   contract-specified prerequisites and paths. Do not add fallback discovery,
   defaults, retries, recovery, auto-recomputation, silent skips, or resilience
   branches; let a required operation fail naturally. Do not add preflight
   gates or check frameworks. Contract-defined scientific conditionals (for
   example, an intrinsic quantity becoming undefined on numerical singularity)
   remain required.
4. Keep `CURRENT.md` as the concise handoff. Scientific definitions and claims
   belong in the active spec; numerical parameters belong in machine-readable
   contracts. Do not duplicate them in policy files or assertion-heavy tests.
5. Preserve unrelated modified and untracked work. Never reset or clean the
   worktree casually.
6. `reference/` and `kaggle/fsq_train_reference.py` are ignored personal
   material. Never inspect, import, package, test, lint, or use them as repo
   evidence.
7. Never store, print, or commit credentials.
8. Kaggle experiments use `scripts/kaggle_kernel.sh` and the existing script
   kernel. New kernels load code from this public repo and mount frozen artifacts
   from Kaggle; never embed a repo payload or add per-kernel builder branches.
   Preserve exact owner/slug/version locators because the authenticated account
   is not necessarily the owner of every input.
9. Long Kaggle jobs are checked only when useful; after confirming `RUNNING`,
   record when to inspect the result and stop polling.
10. `paper/sipaim2026` is the working-paper subtree. Sync it to Overleaf only
   with `scripts/sipaim_overleaf_sync.sh`; never push the whole repository.
11. The separate thesis repository is changed only for a thesis task.
12. Professor-facing GitHub updates are written in Spanish. Do not close an
    issue unless the user asks.

## Scientific Invariants

- The target symmetry is continuous `SO(2)`, not a discrete rotation group.
- Both VAEs are frozen at 60,000 updates. Never train or alter their weights in
  post-hoc geometry experiments.
- Their latent state is posterior `mu/logvar` with shape `(B,16,32,32)`; the
  current geometry uses deterministic posterior `mu` unless stated otherwise.
- The decoder uses raw output for reconstruction geometry; clamping is only for
  image metrics or visualization.
- Sealed-test results cannot select examples, parameters, stopping points,
  thresholds, retries, or architectures.
- Local charts, decoder fibers, pose orbits, quotient classes, geodesics,
  parallel transport, and holonomy are distinct objects. Claim only what the
  numerical experiment actually establishes.
- Compare the two models with the same inputs and numerical budget. Keep
  exploratory, validation, and sealed-test evidence separate.

## Verification

Verification is proportional and read-only:

- parameter/config change: parse it, compile the existing runner, then build and
  validate the kernel;
- numerical formula change: add or run a focused test against a direct formula,
  finite difference, or known synthetic case;
- shared library change: run the affected tests and static checks for the files
  touched;
- CUDA, memory, compilation, and throughput behavior: measure on Kaggle.

Do not run unrelated repository-wide checks and do not auto-format or auto-fix
unrelated files.
