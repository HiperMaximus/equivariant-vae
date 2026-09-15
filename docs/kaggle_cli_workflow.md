# Kaggle CLI Workflow

Status: active
Last updated: 2026-09-14

Kaggle is a remote execution surface. Repository code, contracts and local
receipts are the source of truth; CLI-managed script kernels are the executable
source. Do not use GitHub-linked notebooks.

## Permission policy

Remote permission exists only in the user conversation. Never encode, persist,
validate or consume it with environment variables, contract fields, marker
strings, claim files, receipts, tests or per-Spec branches. Once the user asks
for a remote operation, execute that operation directly. A retry or different
remote mutation still needs conversational scope, but no code-level token.

Receipts are post-operation provenance. They record what Kaggle accepted and
must never act as permission or prevent a later version.

## Commands

Use the generic launcher:

```bash
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
./scripts/kaggle_kernel.sh api-check <kernel-dir>
./scripts/kaggle_kernel.sh push <kernel-dir>
./scripts/kaggle_kernel.sh status-launch <launch-receipt.json>
./scripts/kaggle_kernel.sh logs-launch <launch-receipt.json>
./scripts/kaggle_kernel.sh output-launch <launch-receipt.json> <new-output-dir>
```

`push` validates the package, creates a minimal account-portable snapshot,
submits it, parses Kaggle's returned canonical owner/slug/version and writes a
receipt under `runs/local/kaggle_launches/`. It accepts the version Kaggle
returns; it does not require a fresh slug or version 1.

`output-launch`, `dataset-download` and `pull-launch` require new local
directories to avoid overwriting evidence. This is a filesystem-integrity
check, not a permission check.

## Technical and scientific validation

Keep checks that protect the experiment itself:

- valid metadata and Python source;
- exact dataset/kernel/model locators and owners;
- private/internet/accelerator settings declared by the experiment;
- payload manifests, hashes, regular files and safe archive paths;
- checkpoint, sample, seed and numerical-contract identity;
- finite tensors, JVP/adjoint checks, interval calculations and output schemas;
- separation of validation, sealed test and exploratory evidence.

Large checkpoints and binaries belong in versioned private input datasets.
Generated wrappers validate embedded payloads before importing project code.

## Observability

Long experiments emit flushed, bounded progress records at startup, input and
runtime resolution, each material workload, output publication and terminal
failure/success. Logs may include stage, elapsed time, safe shapes, memory and
exception details needed for diagnosis. Never log credentials, raw data,
checkpoint contents or secret environment values.

Do not wait in-turn through a multi-hour job. After confirming `RUNNING`, give
the user a useful local time to check again. Preserve downloaded outputs under
their exact versioned locator and receipt.
