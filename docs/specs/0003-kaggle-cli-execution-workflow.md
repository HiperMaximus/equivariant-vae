# Spec 0003: Kaggle CLI Execution Workflow

Status: draft active workflow scaffold
Implementation readiness: local scaffold only; not Kaggle-push-ready
Owner/workstream: Kaggle GPU execution and artifact retrieval
Last updated: 2026-06-05

## Purpose

Make Kaggle a controlled remote execution surface for GPU runs while keeping this
repo as the source of truth.

Kaggle must not become a second source of canonical model code. Local repo code,
specs, configs, and launchers define the experiment; Kaggle receives generated or
scaffolded script kernels through the Kaggle API.

## Non-Goals

- Do not use Kaggle as a Git remote.
- Do not require Kaggle's GitHub-linked notebook UI workflow.
- Do not edit the historical FSQ notebooks as the new baseline source.
- Do not push a real training kernel before the behavior inventory and spec 0001
  are locked.
- Do not commit Kaggle credentials, API tokens, output datasets, checkpoints, or
  run artifacts.

## Workflow Contract

The supported workflow is:

```text
repo source -> local Kaggle script kernel folder -> kaggle kernels push
            -> kaggle kernels status/output -> local ignored run artifacts
```

Local commands must go through:

```bash
./scripts/kaggle_kernel.sh
```

Remote writes require explicit user permission plus:

```bash
KAGGLE_PUSH_CONFIRMED=1
```

Remote reads/downloads require explicit user permission plus:

```bash
KAGGLE_REMOTE_CONFIRMED=1
```

Remote pulls that can overwrite local kernel files require explicit user
permission plus:

```bash
KAGGLE_PULL_CONFIRMED=1
```

The current scaffold kernel is:

```text
kaggle/kernels/non_eq_vae_debug
```

It is intentionally not push-ready. The placeholder script exits immediately
until `docs/behavior_inventory_kaggle.md` exists and spec 0001 is locked as
implementation-ready.

## Kaggle Authentication Contract

Kaggle credentials are local user secrets. They must never be printed, stored in
repo files, or committed.

Use the official Kaggle API authentication paths only, such as local CLI login or
the standard local token file. Agents must ask before running any command that
uses network access or remote Kaggle writes.

## Metadata Contract

Each script kernel folder must contain:

- `kernel-metadata.json`
- exactly one declared `code_file`

Metadata should declare:

- `id`
- `title`
- `code_file`
- `language`
- `kernel_type`
- `is_private`
- `enable_gpu`
- `enable_internet`
- `machine_shape`
- `dataset_sources`
- `competition_sources`
- `kernel_sources`
- `model_sources`

Dataset slugs must be explicit. Do not infer them from display names in the
Kaggle web UI.

## Acceptance Criteria

This workflow scaffold is complete when:

1. `docs/kaggle_cli_workflow.md` documents the local workflow;
2. `scripts/kaggle_kernel.sh` validates local metadata and guards remote writes;
3. `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json` exists as a private
   script-kernel scaffold;
4. `kaggle/kernels/non_eq_vae_debug/run.py` is a non-pushable placeholder;
5. preflight tracks the Kaggle workflow files;
6. `runs/` is ignored for downloaded Kaggle outputs;
7. `CURRENT.md` records that the scaffold exists but is not push-ready.

This workflow becomes Kaggle-push-ready only after:

1. `docs/behavior_inventory_kaggle.md` exists;
2. spec 0001 is locked as implementation-ready;
3. real dataset slugs are confirmed;
4. the placeholder guard is removed from `run.py`;
5. the user confirms Kaggle authentication and remote push permission.

## Verification Commands

Local no-network checks:

```bash
./scripts/kaggle_kernel.sh validate
bash -n scripts/kaggle_kernel.sh
python3 -m json.tool kaggle/kernels/non_eq_vae_debug/kernel-metadata.json
./scripts/agent_preflight.sh
```

Remote commands, only after explicit permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
```

## Known Risks

- A Kaggle GitHub-linked notebook can drift from repo code.
- A script kernel can be pushed without the right dataset slugs if metadata is
  guessed from UI display names.
- Pulling from Kaggle can overwrite local kernel files.
- Enabling internet in Kaggle can hide undeclared dependency and code-source
  assumptions.

## Related Files

- `docs/kaggle_cli_workflow.md`
- `scripts/kaggle_kernel.sh`
- `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json`
- `kaggle/kernels/non_eq_vae_debug/run.py`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0002-strict-python-quality-gate.md`
- `docs/equivariant_vae_transition_plan.md`
