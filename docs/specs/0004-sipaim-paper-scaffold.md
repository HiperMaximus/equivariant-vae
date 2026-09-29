# Spec 0004: Working Paper Contract

Status: archived SIPAIM scaffold; Overleaf project migrated to INCISCOS
Owner/workstream: working manuscript
Last updated: 2026-09-29

## Purpose

Keep the manuscript source, bibliography, figures, tables and compiled
advisor-facing PDF together under `paper/sipaim2026`.

## Current State

- SIPAIM 2026 was not submitted and is not an active venue.
- `paper/sipaim2026` remains the local historical scaffold.
- The former SIPAIM Overleaf project was replaced by the curated
  `paper/inciscos2026` review manuscript on 2026-09-29.
- Accepted experiment results are complete in
  `reports/professor/informe_final_experimento_eqvae.{docx,pdf}` but have not
  been transferred into the manuscript.
- A paper update requires a separate user request and venue/format decision.

## Contract

- Historical SIPAIM source: `paper/sipaim2026`.
- Current review source and tracked PDF: `paper/inciscos2026`.
- Overleaf receives only the selected INCISCOS paper files; the old SIPAIM
  subtree pull/push commands are disabled for the migrated project.
- Paper claims must match `CURRENT.md`, Spec 0048 and accepted evidence.
- Validation, sealed-test and exploratory evidence remain explicitly separated.
- No universal superiority claim is permitted.
- Figures and tables must reuse accepted results rather than rerun experiments.

## Verification For An Authorized Paper Update

```bash
./scripts/sipaim_overleaf_sync.sh check
./scripts/sipaim_overleaf_sync.sh compile
git diff --check
```

Remote Overleaf reads, pulls and pushes require explicit permission and
`OVERLEAF_SYNC_CONFIRMED=1`. Do not push the whole repository.

## Current Boundary

This SIPAIM scaffold is archived. The INCISCOS manuscript is the current
professor-review paper.
