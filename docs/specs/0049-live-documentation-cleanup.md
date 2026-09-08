# Spec 0049: Live Documentation Cleanup

Status: implemented and verified
Owner/workstream: repository state and handoff hygiene
Last updated: 2026-09-08

## Purpose

Remove stale planning state and chronological narration from live repository
documents after completion of the experiment and professor report.

## Scope

- Keep `CURRENT.md` as a compact current-state handoff, not a run diary.
- Keep `GOAL.md`, the architecture reference, dataset contract and Kaggle
  workflow limited to durable current facts.
- Delete completed entries from `docs/open_follow_ups.md` and retain only
  actionable current debt.
- Make the spec index own completed lifecycle state while preserving exact
  hash-bound spec bytes.
- Preserve exact resource identities, scientific results, consumed-authority
  boundaries, verifier literals and paths used by code.

## Non-Goals

- No experiment, model, data, report, paper or thesis change.
- No Kaggle, GitHub or Overleaf call.
- No deletion of evidence or executable artifacts merely because their run is
  complete.

## Acceptance Criteria

1. Live landing documents contain no execution chronology or stale next steps.
2. `CURRENT.md` records only the current result, evidence, blockers and next
   authorized boundary.
3. Required files and guard literals used by `scripts/kaggle_kernel.sh` remain.
4. The repository and workspace preflights pass.
5. `git diff --check` passes.

## Verification

```bash
./scripts/agent_preflight.sh
git diff --check
```

Repository preflight, `git diff --check`, and focused Ruff/format checks pass.
No experiment, report, paper or remote artifact changed.
