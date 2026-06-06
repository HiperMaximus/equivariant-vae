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
| `docs/spec_driven_development.md` | Spec-first workflow rules. | The spec workflow changes. |
| `docs/specs/README.md` | Spec status index and implementation-readiness gate. | A spec is added, locked, implemented, blocked, or superseded. |
| `docs/specs/` | Workstream-level implementation contracts. | A feature, experiment, evaluation, paper artifact, or workflow contract changes. |
| `pyproject.toml` | Python environment, Ruff, and BasedPyright contract. | Tooling, dependencies, or Python target changes. |
| `uv.lock` | Resolved repo-local Python environment. | `pyproject.toml` dependencies or dependency groups change. |
| `docs/issue_image_inventory.md` | Evidence from inspected issue images. | New issue images are inspected or requirements are corrected. |

Adapters and summaries should point to canonical files instead of copying long
rule lists. If duplication is necessary for safety, keep the list identical.

Do not use root `requirements.txt` as a state file. If a pip requirements export
is needed for Kaggle, make it generated, context-specific, and documented in the
relevant spec.

## Handoff Memory Rule

Before an agent hands work back, pauses, or leaves a partial implementation, it
must update the repo handoff memory. At minimum:

- update `CURRENT.md` with what changed, what is currently in progress, exactly
  where the agent left off, the next concrete action, blockers, and verification
  status;
- update active plans or specs when their task order, blockers, acceptance
  criteria, or implementation-readiness state changed;
- update `README.md`, `AGENTS.md`, or `GOAL.md` only when stable workflow rules,
  boundaries, or north-star requirements changed;
- delete or replace stale statements rather than leaving contradictory historical
  breadcrumbs.

The goal is that a fresh Codex or Claude session can resume from files, not from
chat memory.

## Canonical Memory Rule

Adversarial subagent checking is part of the repo workflow. Future agents should
use it before major workflow, architecture, evaluation, or paper-claim changes
when subagent tooling is available.

If subagents are unavailable, perform the same audit manually and say that the
independent clean-context pass could not be run.
