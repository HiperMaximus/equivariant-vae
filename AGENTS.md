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
   `docs/kaggle_cli_workflow.md`,
   `docs/behavior_inventory_kaggle.md`,
   `docs/overleaf_sync_workflow.md`,
   `docs/agentic_review_workflow.md`,
   `docs/spec_driven_development.md`, `docs/specs/README.md`, active specs
   linked from that index, and `docs/decisions/README.md`.
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
17. Before handing work back or stopping at a partial state, update the repo
    memory/handoff files. At minimum, record what changed, what is currently in
    progress, exactly where the agent left off, the next concrete action,
    blockers, and verification status in `CURRENT.md` and any affected active
    plan/spec.
18. Claude-specific instructions live in `CLAUDE.md` but are adapters only.
    Canonical facts belong in `AGENTS.md`, `CURRENT.md`, `GOAL.md`, and docs.
19. Before substantial work, run:

   ```bash
   ./scripts/agent_preflight.sh
   ```
20. For substantial workflow, architecture, evaluation, or paper-claim changes,
    use independent clean-context adversarial subagent reviews when the tooling
    is available. Follow `docs/agentic_review_workflow.md`.
21. Use spec-driven development for substantial implementation, experiment,
    evaluation, paper, or workflow changes. Write or update the relevant spec in
    `docs/specs/` before coding, then verify against its acceptance criteria.
22. For Python changes, run `./scripts/python_quality.sh` before handing work
    back. The script intentionally uses the existing repo-local `.venv` and does
    not run `uv sync` or download dependencies. If the environment needs to be
    created or refreshed, ask the user first, then run
    `uv sync --locked --python 3.12 --group dev`.
23. Python quality is intentionally strict: Ruff selects `ALL`, BasedPyright is
    strict, no global ignores are allowed, and tests may ignore only Ruff `S101`
    for bare `assert`.
24. Local repo tests use CPU-only PyTorch. GPU training belongs to Kaggle.
25. Python dependency truth lives in `pyproject.toml` for direct dependencies and
    `uv.lock` for the resolved local environment. A root `requirements.txt` is
    not allowed; pip requirements files may only be generated, context-specific
    exports such as a future Kaggle bootstrap file.
26. Kaggle is a remote execution surface, not a Git remote. Use
    `./scripts/kaggle_kernel.sh` and `docs/kaggle_cli_workflow.md`; do not use a
    GitHub-linked Kaggle notebook as the source of truth. Kaggle remote writes
    require explicit user permission and `KAGGLE_PUSH_CONFIRMED=1`.

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
- `docs/kaggle_cli_workflow.md` for CLI-managed Kaggle script kernels.
- `docs/behavior_inventory_kaggle.md` for historical Kaggle data, training,
  resume, metric, and artifact behavior.
- `docs/overleaf_sync_workflow.md` for the full workflow and failure modes.
- `docs/decisions/README.md` for settled project decisions.
- `docs/agentic_review_workflow.md` for independent adversarial review.
- `docs/spec_driven_development.md`, `docs/specs/README.md`, and active specs
  in `docs/specs/` for implementation contracts.
- `docs/specs/0002-strict-python-quality-gate.md` for Python quality rules.
