# Decision 0003: Overleaf Subtree And Tracked PDF

Status: active
Date: 2026-06-05

## Decision

The SIPAIM paper source remains in this repo at:

```text
paper/sipaim2026
```

Overleaf receives only that subtree. The repo intentionally tracks the current
advisor-facing PDF at:

```text
paper/sipaim2026/sipaim2026.pdf
```

## Rationale

The professor needs to view and edit the paper in Overleaf, while the repo needs
to keep the paper text, source assets, workflow, and current PDF together with
the experiment code and requirements.

## Consequences

- Do not push the full repo to Overleaf.
- Do not use `git push overleaf` directly.
- Use `scripts/sipaim_overleaf_sync.sh` for setup, compile, pull, and push.
- Refresh `sipaim2026.pdf` before pushing paper changes or reporting that the
  paper is current.
