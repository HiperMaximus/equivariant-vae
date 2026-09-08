# Spec 0004: Working Paper Contract

Status: scaffold implemented; result integration not authorized
Owner/workstream: working manuscript
Last updated: 2026-09-08

## Purpose

Keep the manuscript source, bibliography, figures, tables and compiled
advisor-facing PDF together under `paper/sipaim2026`.

## Current State

- SIPAIM 2026 was not submitted and is not an active venue.
- The existing IEEE manuscript is a working scaffold, not the final professor
  report.
- Accepted experiment results are complete in
  `reports/professor/informe_final_experimento_eqvae.{docx,pdf}` but have not
  been transferred into the manuscript.
- A paper update requires a separate user request and venue/format decision.

## Contract

- Source of record: `paper/sipaim2026`.
- Tracked compiled artifact: `paper/sipaim2026/sipaim2026.pdf`.
- Overleaf receives only this subtree through
  `scripts/sipaim_overleaf_sync.sh`.
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

No paper or Overleaf mutation is authorized. The professor-facing report is the
current presentation artifact.
