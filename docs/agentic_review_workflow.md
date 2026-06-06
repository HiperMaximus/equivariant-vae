# Agentic Review Workflow

Status: active workflow
Last updated: 2026-06-05

Use adversarial subagent reviews for important changes where future mistakes
would be expensive, especially:

- repo memory, instructions, workflow, or Overleaf sync changes;
- architecture decisions for the comparable VAE and `SO(2)` steerable model;
- evaluation requirements derived from GitHub issues or issue images;
- paper claims, figure requirements, and advisor-facing updates;
- any change that could confuse the thesis repo, paper repo, or Overleaf
  subtree boundaries.

Small local edits do not need subagents. The point is to catch hidden
coordination failures before they harden into project memory.

## Required Pattern

For substantial planning or workflow changes:

1. Spawn independent clean-context subagents.
   - Do not fork the full current chat unless the task truly needs it.
   - Give each subagent a narrow adversarial brief.
   - Prefer read-only explorers for audits.
2. Split the audit angles.
   - Workspace/repo boundary safety.
   - Instruction and memory consistency.
   - Script/tooling safety and failure modes.
   - Requirements coverage against issues and issue images.
   - Stale/historical artifact detection.
3. Ask for concrete findings only.
   - File path and line or section when possible.
   - Severity.
   - Why it can mislead a future agent.
   - Recommended fix.
4. Integrate the findings locally.
   - Apply only fixes that improve the canonical workflow.
   - Delete or replace stale information instead of appending contradictions.
5. Re-run local checks.
   - `./scripts/agent_preflight.sh`
   - targeted stale-term searches for the changed docs.
   - paper compile or Overleaf checks only when relevant.
6. Record the outcome.
   - Update `CURRENT.md` when the active state, blockers, or next steps change.
   - Update this workflow if the review process itself changes.

## State File Policy

Keep each project fact in the smallest stable home:

| File | Purpose | Update when |
| --- | --- | --- |
| `AGENTS.md` | Hard rules for agents and repo boundaries. | A rule, safety constraint, or required reading sequence changes. |
| `CLAUDE.md` | Claude adapter only. | The canonical landing sequence changes. Do not add independent facts. |
| `CURRENT.md` | Active handoff state, blockers, and next concrete steps. | Workstream, blockers, or next actions change. |
| `GOAL.md` | Durable repo north star and must-not-lose requirements. | Research scope or non-negotiable deliverables change. |
| `README.md` | Stable human onboarding. | Stable workflow entry points or repo purpose changes. |
| `docs/decisions/` | Settled decisions and their rationale. | A decision is made, superseded, or explicitly reopened. |
| `docs/*_plan*.md` | Active implementation plan/checklists. | Execution phases, gates, or technical assumptions change. |
| `docs/issue_image_inventory.md` | Evidence from inspected issue images. | New issue images are inspected or requirements are corrected. |

Adapters and summaries should point to canonical files instead of copying long
rule lists. If duplication is necessary for safety, keep the list identical.

## Canonical Memory Rule

Adversarial subagent checking is part of the repo workflow. Future agents should
use it before major workflow, architecture, evaluation, or paper-claim changes
when subagent tooling is available.

If subagents are unavailable, perform the same audit manually and say that the
independent clean-context pass could not be run.
