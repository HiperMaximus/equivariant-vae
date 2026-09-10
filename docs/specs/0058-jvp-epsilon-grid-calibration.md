# Spec 0058: JVP Epsilon-Grid Calibration

Status: locked / implementation-ready
Owner/workstream: Spec 0053 numerical calibration
Last updated: 2026-09-10

## Purpose

Spec 0057's sole private parent completed frozen-bundle loading and eight
primary plus sixteen diagnostic JVP finite-difference records for one blind
branch. Its fixed primary `epsilon=0.001` failed, while diagnostics at `0.004`
and `0.008` passed. This separate, minimal calibration measures a prespecified
epsilon grid on both anonymous branches before any later scientific geometry
experiment adopts a finite-difference reference step.

It is not a scientific model comparison, full preflight, tolerance relaxation,
training run, checkpoint selection, or retry of Spec 0057. It has no child and
does not authorize any scientific workload after completion.

## Locked Protocol

The immutable contract is
`docs/data/spec0058_jvp_epsilon_grid_calibration_contract.json`. On the same
rank-zero fixed patch, load the same frozen bundle and randomly permute models
under non-persisted labels `branch_a` and `branch_b`. For each branch and each
of the eight fixed dense RMS-one Rademacher directions, compute the autodiff
JVP once and compare it with a central finite difference at exactly
`epsilon in {0.002, 0.004, 0.008, 0.016}`. Model mechanics remain FP32;
detached residual reductions are CPU FP64.

For each epsilon, report anonymous per-branch and global aggregate/worst
relative-L2 residuals. The selection is mechanical: choose the smallest grid
epsilon whose global aggregate **and** worst residual are at most `0.005`; if
none qualifies, emit `no_eligible_epsilon`. This selection is a calibration
artifact only and does not pass or fail either VAE.

The wrapper performs no HVP, VJP, compilation, continuous rotation, geodesic,
or full preflight workload. It emits only flushed, closed-schema, result-blind
JSONL progress. The frozen-bundle-ready event precedes model loading. Public
files and logs expose neither paths, hashes, filenames, model identities,
permutation, raw inputs, nor checkpoint identities.

Binary search is forbidden: finite-difference error combines rounding and
truncation effects and is not assumed monotone. The fixed log-spaced grid is
sufficient for a robust reference step; it does not seek a globally optimal
epsilon.

## Remote Boundary And Verification

Guard: `spec0058_jvp_epsilon_grid_calibration_authorized`. Exactly one private
T4, no-internet parent is permitted at
`maximshtefan/eqvae-jvp-epsilon-grid-calibration-05a08ab5`. It must use the
two named input datasets. No wait, automatic retry, child, output-driven rule
change, or scientific model claim is authorized.

Before push: focused unit tests, Ruff, BasedPyright, embedded build/validate/
check, Bash syntax, `agent_preflight.sh`, `git diff --check`, and independent
numerical, observability, and provenance reviews. After one status confirms
`RUNNING`, stop; download terminal output only when explicitly checked.
