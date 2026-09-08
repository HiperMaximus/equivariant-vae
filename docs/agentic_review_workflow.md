# Agentic Review Workflow

Status: active
Last updated: 2026-09-08

Use independent clean-context review when a mistake would materially affect
architecture, workflow, evaluation, paper claims, repository state or external
systems.

## Review Pattern

1. Give each reviewer a narrow, read-only brief.
2. Split review angles: scientific claims, implementation correctness,
   provenance, workflow safety, layout/accessibility and stale-state detection.
3. Require findings with severity, exact location, impact and recommended fix.
4. Verify each finding against source evidence before applying it.
5. Apply only in-scope fixes, then rerun relevant checks.
6. Record only the resulting current state in `CURRENT.md` or the active spec.

If independent reviewers are unavailable, perform the same checks manually and
state that no independent pass occurred.

## Working-Tree Safety

- Reviewers inspecting uncommitted work must not modify tracked files.
- Reason about mutations or use a disposable scratch copy; never mutate the
  shared worktree to test a hypothesis.
- A separate Git worktree cannot review uncommitted changes from the original
  checkout unless those changes are explicitly transferred.
- After review, inspect the actual diff against intent and rerun the guarding
  tests locally.

## State Files

| File | Single responsibility |
| --- | --- |
| `AGENTS.md` | Binding rules and repository boundaries |
| `CLAUDE.md` | Thin adapter to canonical files |
| `CURRENT.md` | Current outcome, blockers and next authorized boundary |
| `GOAL.md` | Durable scientific goal and invariants |
| `README.md` | Human onboarding and workflow entry points |
| `docs/repo_goal_and_requirements.md` | Professor requirements, metrics and claim gates |
| `docs/issue_image_inventory.md` | Requirements derived from inspected issue images |
| `docs/equivariant_vae_transition_plan.md` | Frozen architecture/fairness contract |
| `docs/behavior_inventory_kaggle.md` | Current data/source behavior contract |
| `docs/kaggle_cli_workflow.md` | Remote execution and authorization workflow |
| `docs/specs/README.md` | Current spec lifecycle index |
| `docs/specs/` | Detailed implementation/evidence contracts |
| `docs/decisions/` | Settled decisions still governing the repo |
| `docs/open_follow_ups.md` | Unresolved actionable debt only |
| `pyproject.toml`, `uv.lock` | Direct and resolved Python dependency truth |

Do not duplicate exact results or operational status across several files. Link
to the smallest authoritative home.

## Handoff Gate

Before handoff:

- update `CURRENT.md` only if outcome, blocker or next boundary changed;
- update affected specs and indexes;
- delete completed to-dos and stale narration;
- run `./scripts/agent_preflight.sh`;
- run focused checks and `git diff --check`;
- preserve unrelated worktree changes.
