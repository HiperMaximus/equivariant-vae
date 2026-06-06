# Overleaf Sync Workflow

This repo keeps the SIPAIM paper in `paper/sipaim2026` so the current text and
PDF can be visible in GitHub. The same folder is synced to Overleaf for advisor
review.

The active manuscript should not be moved to a separate local folder. The source
of record is `paper/sipaim2026`, and the professor-facing Overleaf project is a
synced copy of that subtree.

## Repositories

```text
equivariant-vae GitHub repo
  origin: https://github.com/HiperMaximus/equivariant-vae.git
  paper:  paper/sipaim2026

Overleaf project
  web:    https://www.overleaf.com/project/69c614433cbc9e46cf226d24
  remote: https://git.overleaf.com/69c614433cbc9e46cf226d24
  branch: master
```

Only `paper/sipaim2026` should go to Overleaf. The rest of the repo contains
code, experiment plans, notebooks, and implementation details that do not belong
in the Overleaf project.

## One-Time Setup

From the repo root:

```bash
./scripts/sipaim_overleaf_sync.sh setup
```

This configures a local Git remote named `overleaf`.

The script refuses embedded credentials in remote URLs. Use a Git credential
helper for Overleaf tokens.

Overleaf pull/push operations require explicit user permission. After checking
status and receiving permission, run them with:

```bash
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

## Normal Local-to-Overleaf Flow

Use this when local paper edits should appear in Overleaf.

```bash
./scripts/sipaim_overleaf_sync.sh check
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull

# Edit files under paper/sipaim2026.
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Update SIPAIM paper"

OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

Why this order:

- `pull` first reduces conflicts with advisor edits from Overleaf.
- `compile` creates the advisor-facing `sipaim2026.pdf`.
- `commit` is required because subtree push exports committed history, not
  unsaved working-tree changes.
- `push` exports only `paper/sipaim2026` to Overleaf.

## Normal Overleaf-to-Local Flow

Use this after the professor edits in Overleaf.

```bash
./scripts/sipaim_overleaf_sync.sh check
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Pull Overleaf paper edits"
```

If the pull creates merge conflicts, resolve only files under
`paper/sipaim2026`, compile again, then commit.

## Important Warnings

- Never run `git push overleaf` from this repo.
- Never set `overleaf` as `origin`.
- Never embed Overleaf tokens in Git remote URLs.
- Never run Overleaf pull/push as an agent without asking the user first.
- Never push notebooks, training code, or experiment folders to Overleaf.
- Avoid concurrent editing: if the professor is actively editing in Overleaf,
  pull before making local edits and push only after checking with them.
- Overleaf uses `master`; do not expect normal branch workflows there.
- Overleaf comments/track changes can be awkward with Git pushes. Prefer normal
  text edits in Overleaf when using this workflow.

## Generated Files

Track:

```text
paper/sipaim2026/sipaim2026.pdf
```

This PDF is intentionally tracked so the latest compiled paper is visible in the
GitHub repo and in Overleaf. Refresh it with:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

Ignore:

```text
paper/sipaim2026/main.pdf
paper/sipaim2026/*.aux
paper/sipaim2026/*.log
paper/sipaim2026/*.bbl
paper/sipaim2026/*.fls
paper/sipaim2026/*.fdb_latexmk
```

The `sipaim2026.pdf` file is tracked because it is useful for GitHub/advisor
visibility. Build artifacts remain ignored.
