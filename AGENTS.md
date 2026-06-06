# Repository Instructions

This repo is the paper/research repository for the equivariant VAE work.

## Project Boundaries

- Main thesis repo:
  `/home/maximus/Documents/Tesis/Tesis`
- This repo:
  `/home/maximus/Documents/Tesis/equivariant-vae`
- SIPAIM paper subtree:
  `paper/sipaim2026`
- Overleaf Git remote:
  `https://git.overleaf.com/69c614433cbc9e46cf226d24`

## Hard Rules

1. Before architecture, evaluation, paper, or workflow changes, read the
   canonical landing sequence:
   `AGENTS.md`, `CURRENT.md`, `GOAL.md`,
   `docs/repo_goal_and_requirements.md`,
   `docs/issue_image_inventory.md`,
   `docs/equivariant_vae_transition_plan.md`,
   `docs/overleaf_sync_workflow.md`,
   `docs/agentic_review_workflow.md`, and `docs/decisions/README.md`.
2. Do not push this whole repo to Overleaf.
3. Do not add Overleaf as `origin`.
4. Do not run plain `git push overleaf`.
5. Sync Overleaf only through:

   ```bash
   ./scripts/sipaim_overleaf_sync.sh
   ```

6. The active paper source lives in `paper/sipaim2026`.
7. The tracked advisor-facing PDF should be `paper/sipaim2026/sipaim2026.pdf`.
   Keep it updated so the current paper can be viewed from both GitHub and
   Overleaf. Keep `main.pdf`, logs, aux files, and other LaTeX build artifacts
   ignored.
8. Pull Overleaf edits before local paper edits when the professor may have
   changed the project.
9. Commit local paper changes before pushing the subtree to Overleaf.
   Overleaf `pull` and `push` require explicit user permission and must be run
   with `OVERLEAF_SYNC_CONFIRMED=1` only after that permission.
10. The architecture target is continuous `SO(2)` steerability, not a
   discrete-group implementation.
11. GitHub issue updates intended for the thesis professor should be written in
    Spanish unless the user asks otherwise.
12. GitHub issue updates should say what changed, where it lives, and what is
    still pending. Do not close issues unless the user explicitly asks.
13. Never store, print, or commit Overleaf tokens or other credentials.
14. GitHub issue images are requirements evidence. Inspect them before deriving
    plans, figures, metrics, or deliverables from issue comments.
15. Keep `README.md`, `GOAL.md`, `AGENTS.md`, plans, and workflow docs current.
    Delete or replace stale/bad/incorrect information instead of appending
    contradictory historical notes. Use the state-file policy in
    `docs/agentic_review_workflow.md`.
16. Keep `CURRENT.md` updated after meaningful shifts in active work, blockers,
    or next steps.
17. Claude-specific instructions live in `CLAUDE.md` but are adapters only.
    Canonical facts belong in `AGENTS.md`, `CURRENT.md`, `GOAL.md`, and docs.
18. Before substantial work, run:

   ```bash
   ./scripts/agent_preflight.sh
   ```
19. For substantial workflow, architecture, evaluation, or paper-claim changes,
    use independent clean-context adversarial subagent reviews when the tooling
    is available. Follow `docs/agentic_review_workflow.md`.

## Safe Paper Workflow

```bash
./scripts/sipaim_overleaf_sync.sh check
./scripts/sipaim_overleaf_sync.sh setup
OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh pull

# edit paper/sipaim2026
./scripts/sipaim_overleaf_sync.sh compile
git add paper/sipaim2026
git commit -m "Update SIPAIM paper"

OVERLEAF_SYNC_CONFIRMED=1 ./scripts/sipaim_overleaf_sync.sh push
```

See:

- `CURRENT.md` for active status and next concrete steps.
- `GOAL.md` for the repo north star.
- `docs/repo_goal_and_requirements.md` for issue-derived deliverables.
- `docs/issue_image_inventory.md` for inspected issue screenshots.
- `docs/overleaf_sync_workflow.md` for the full workflow and failure modes.
- `docs/decisions/README.md` for settled project decisions.
- `docs/agentic_review_workflow.md` for independent adversarial review.
