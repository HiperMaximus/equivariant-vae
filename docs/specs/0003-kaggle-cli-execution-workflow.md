# Spec 0003: Kaggle CLI Execution Workflow

Status: draft active workflow scaffold
Implementation readiness: narrow capped smoke push path ready after permission;
full benchmark/full-run launchers are not Kaggle-push-ready
Owner/workstream: Kaggle GPU execution and artifact retrieval
Last updated: 2026-06-06

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
- Do not push a full training or runtime-benchmark kernel before the spec 0001
  launcher is implemented and locally verified. The only current exception is
  the capped `kaggle_smoke_ready` debug script, which runs at most three train
  steps and one clean-validation batch and writes non-promotable smoke evidence.
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

Spec 0001 implementation must add a build step that copies the repo code/config
payload needed by Kaggle into the kernel folder:

```bash
./scripts/kaggle_kernel.sh build
```

The generated `kaggle/kernels/*/payload/` directory is ignored and must not be
committed. It is rebuilt from `src/eqvae`, `configs/spec0001`, `pyproject.toml`,
and `uv.lock`. Kaggle kernels must not resolve/install dependencies from that
metadata unless a later spec explicitly introduces an offline wheel/bootstrap
path.

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

It now contains the narrow capped smoke launcher only. It is push-ready only for
the `kaggle_smoke_ready` debug smoke after explicit user permission,
`KAGGLE_PUSH_CONFIRMED=1`, and a rebuilt payload. It is not a full benchmark or
full-run launcher.

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

The first confirmed training dataset source is:

```text
maximusshtefan/patches-pre-shuffled-ubc-ocean
```

Other historical sources are recorded in
`docs/behavior_inventory_kaggle.md`.

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

This workflow becomes Kaggle-push-ready for the narrow capped smoke only after:

1. spec 0001 and the spec index contain `kaggle_smoke_ready`;
2. the smoke script kernel has `KAGGLE_SMOKE_READY = True`;
3. `./scripts/kaggle_kernel.sh build` has copied the current `src/eqvae`,
   `configs/spec0001`, `pyproject.toml`, and `uv.lock` into the ignored payload;
4. local smoke tests and the production Python quality gate pass;
5. `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes the
   read-only checks or records only the known quota/files warnings;
6. the user explicitly approves the remote write/run.

The 2026-06-13 real-data smoke push showed that even a three-step script can
spend substantial time in Kaggle setup when it attaches the 60 GB+
`patches-pre-shuffled-ubc-ocean` dataset. Before the next setup-only remote
test, add a distinct synthetic setup-smoke path: no `dataset_sources`, no real
dataset attachment, tiny synthetic UBC-format shards generated under
`/kaggle/working`, and a separate non-promotable status/source. Keep the
current dataset-source guard for real-data smoke and benchmark kernels.

The full benchmark/full-run workflow becomes Kaggle-push-ready only after:

1. spec 0001 is locked as implementation-ready;
2. the spec 0001 code/config payload is built into the kernel folder;
3. a full benchmark/full-run launcher replaces the capped smoke launcher;
4. local spec 0001 verification passes;
5. for benchmark kernels, metadata validation requires
   `machine_shape == "NvidiaTeslaT4"` and the safe `single_visible_t4` versus
   `dual_t4_ddp` launch mode recorded in `docs/kaggle_cli_workflow.md`;
6. `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes the
   required read-only auth/list/status/logs/dataset checks; if the quota
   endpoint warns, GPU quota is verified in the Kaggle web UI and recorded in
   the run notes;
7. the user confirms Kaggle authentication and remote push permission.

## Verification Commands

Current scaffold no-network checks:

```bash
./scripts/kaggle_kernel.sh validate
bash -n scripts/kaggle_kernel.sh
python3 -m json.tool kaggle/kernels/non_eq_vae_debug/kernel-metadata.json
./scripts/agent_preflight.sh
```

Spec 0001 post-implementation payload check:

```bash
./scripts/kaggle_kernel.sh build
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
