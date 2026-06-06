# Claude Repository Instructions

This file is a thin adapter for Claude. The canonical repository instructions
live in `AGENTS.md`, `CURRENT.md`, `GOAL.md`, and the docs listed below.

Read the canonical landing sequence before architecture, evaluation, paper,
workflow, or Overleaf work:

1. `AGENTS.md`
2. `CURRENT.md`
3. `GOAL.md`
4. `docs/repo_goal_and_requirements.md`
5. `docs/issue_image_inventory.md`
6. `docs/equivariant_vae_transition_plan.md`
7. `docs/overleaf_sync_workflow.md`
8. `docs/decisions/README.md`
9. `docs/agentic_review_workflow.md`
10. `docs/spec_driven_development.md`
11. active specs in `docs/specs/`

For Python changes, run `./scripts/python_quality.sh` before finalizing. Ask the
user first if `uv` needs network access to sync Python or dependencies.

Run this preflight before substantial work:

```bash
./scripts/agent_preflight.sh
```

Do not duplicate project facts here. Update the canonical files instead.
