# Spec-Driven Development Workflow

Status: active workflow
Last updated: 2026-06-05

This repo should use spec-driven development for meaningful code, experiment,
evaluation, paper, and workflow changes.

The pattern is:

```text
spec -> adversarial review when stakes are high -> implementation -> verification -> spec/docs refresh
```

The goal is not bureaucracy. The goal is to make future agents agree on what
"done" means before code or claims start drifting.

## When A Spec Is Required

Write or update a spec before:

- implementing model architecture changes;
- changing the data contract, split policy, training objective, or latent target;
- adding evaluation metrics, figures, dashboards, or paper artifacts;
- changing Overleaf/GitHub workflow behavior;
- making paper claims or issue updates that depend on experiment results;
- touching shared infrastructure such as configs, logging, checkpointing,
  launchers, or artifact writers.

Tiny mechanical edits do not need a new spec, but they must still keep existing
specs accurate.

## Spec Location

Specs live in:

```text
docs/specs/
```

Use `docs/specs/template.md` for new specs.

Active specs should be linked from `CURRENT.md` or the relevant plan. Completed
or superseded specs should be updated in place with their final status, not left
as misleading drafts.

## Minimum Spec Contents

Every implementation spec should include:

- status and owner/workstream;
- problem statement;
- non-goals;
- inputs, outputs, and data contracts;
- architecture or workflow contract;
- config knobs and defaults;
- acceptance artifacts;
- tests and verification commands;
- known risks and adversarial checks;
- open questions that block implementation;
- links to related issues, issue images, decisions, plans, and paper sections.

Experiment specs must additionally define:

- dataset split policy;
- corruption/augmentation policy;
- metrics with sample count `n`;
- qualitative artifact protocol;
- parameter/compute reporting policy;
- seed and tuning budget policy.

Paper specs must additionally define:

- section or figure/table target;
- source files;
- expected PDF refresh behavior;
- Overleaf sync rule;
- advisor-facing issue update rule.

## Workflow

1. Create or refresh the spec.
2. Check it against `GOAL.md`, `docs/repo_goal_and_requirements.md`,
   `docs/issue_image_inventory.md`, and `docs/decisions/`.
3. For substantial or risky work, run clean-context adversarial subagent review
   on the spec before implementation.
4. Implement only what the spec covers.
5. Verify against the spec's acceptance criteria.
6. For Python changes, run `./scripts/python_quality.sh` after implementation.
7. Update the spec, `CURRENT.md`, and any affected plan/readme files.
8. Delete or replace stale information. Do not leave contradictory historical
   notes.

## Memory Rule

Spec-driven development is part of the repo workflow. Future agents should not
start substantial implementation from chat context alone; they should first
write or update the relevant spec in `docs/specs/`.
