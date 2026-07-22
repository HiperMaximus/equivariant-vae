# Claude Repository Instructions

This file is a thin adapter for Claude. The canonical repository instructions
live in `AGENTS.md`, `CURRENT.md`, `GOAL.md`, and the docs listed below.

Read the canonical landing sequence before architecture, evaluation, paper,
workflow, or Overleaf work. `AGENTS.md` is the BINDING hard-rules contract (rules
1–31) — read it in full, do not skim: agents keep relapsing into already-corrected
traps (especially rule 15 lean-and-live, rule 29 verify-the-premise, rule 30
speed-first, rule 22 detached gate).

1. `AGENTS.md`
2. `CURRENT.md`
3. `GOAL.md`
4. `docs/repo_goal_and_requirements.md`
5. `docs/issue_image_inventory.md`
6. `docs/equivariant_vae_transition_plan.md`
7. `docs/kaggle_cli_workflow.md`
8. `docs/behavior_inventory_kaggle.md`
9. `docs/overleaf_sync_workflow.md`
10. `docs/agentic_review_workflow.md`
11. `docs/spec_driven_development.md`
12. `docs/specs/README.md`
13. active specs linked from `docs/specs/README.md`
14. `docs/decisions/README.md`

For Python changes, run `./scripts/python_quality.sh` before finalizing. The
script uses the existing `.venv` and does not sync dependencies. Ask the user
first before running `uv sync --locked --python 3.12 --group dev`.

Run this preflight before substantial work:

```bash
./scripts/agent_preflight.sh
```

Do not duplicate project facts here. Update the canonical files instead.
