# Kaggle CLI Workflow

Status: draft workflow scaffold
Last updated: 2026-06-10

Kaggle is a remote execution surface, not a Git remote. This repo remains the
source of truth for experiment code, specs, configs, and paper-facing claims.

## Current State

Historical Kaggle notebooks live in:

```text
kaggle/train_runs
kaggle/dataset_generation
```

They are JSON notebooks kept as historical evidence and behavior-inventory input.
Do not edit them into the new baseline.

The first CLI-managed script-kernel scaffold lives in:

```text
kaggle/kernels/non_eq_vae_debug
```

It is not push-ready yet. It intentionally exits until spec 0001 is locked as
implementation-ready and the real launcher replaces the placeholder.

The behavior inventory now lives at:

```text
docs/behavior_inventory_kaggle.md
```

On 2026-06-06, `./scripts/kaggle_kernel.sh check` passed on this laptop with
Kaggle CLI 2.2.1. Authentication is still a user-local secret and should be
rechecked before remote reads or writes.

## Local Commands

Validate the local scaffold:

```bash
./scripts/kaggle_kernel.sh validate
```

Check whether the Kaggle CLI is installed and whether local metadata is valid:

```bash
./scripts/kaggle_kernel.sh check
```

Push a script kernel only after explicit user permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
```

Check remote status after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
```

Download outputs into ignored local run artifacts after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
```

Pulling from Kaggle can overwrite local files and requires explicit permission:

```bash
KAGGLE_PULL_CONFIRMED=1 ./scripts/kaggle_kernel.sh pull
```

## Credentials

Kaggle credentials are local secrets. Do not store, print, or commit them.

The official Kaggle API supports local CLI authentication and the standard local
token file. Agents must ask before running network commands or remote writes.

## Dataset Sources

Attach Kaggle datasets through `kernel-metadata.json`, not by hard-coding UI
display names in the script.

Use exact dataset slugs, for example:

```json
"dataset_sources": ["owner/dataset-slug"]
```

The current scaffold uses the confirmed pre-shuffled patch dataset:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
```

Other confirmed historical slugs are recorded in
`docs/behavior_inventory_kaggle.md`. Do not attach
`maximusshtefan/non-eq-vae-output` to the new baseline unless intentionally
reproducing the old FSQ resume path.

The pre-shuffled patch dataset is the confirmed train/validation patch source.
It contains `ubc_train_shuffled.*` and `ubc_ocean_valid.*`, but no held-out test
shard. Final evaluation needs a separate sealed test dataset/source.

The push wrapper refuses remote writes while `dataset_sources` is empty, while
the placeholder guard remains, or while spec 0001 is not locked as
implementation-ready.

## GitHub Linking

Kaggle's web UI can show a notebook as linked from GitHub, but that is not the
workflow here. For agentic work, the repo should generate or own the script
kernel folder, and the Kaggle API should upload that folder.

If someone edits a kernel in the Kaggle UI, pull it locally, inspect the diff,
and reconcile it into the repo. Do not let UI edits become the source of truth.

## Official References

- Kaggle API README: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- Kaggle kernel commands: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md
- Kaggle kernel metadata: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
