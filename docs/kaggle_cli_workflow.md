# Kaggle CLI Workflow

Status: active operational contract
Last updated: 2026-09-08

Kaggle is a remote execution surface, not a Git remote. Repository code, specs,
configs and local receipts are the source of truth. `CURRENT.md` records whether
a job is active; currently none is active and all completed evaluation
one-shot authorities are consumed.

## Entry Point

Use only:

```bash
./scripts/kaggle_kernel.sh help
./scripts/kaggle_kernel.sh build <kernel-dir>
./scripts/kaggle_kernel.sh validate <kernel-dir>
./scripts/kaggle_kernel.sh check <kernel-dir>
```

The script's `help` output is the canonical action inventory. Kernel sources
live under `kaggle/kernels/`; generated `run.py` files and payloads are ignored.
Do not use a GitHub-linked notebook as executable source.

## Authorization

Local build, validation and preflight commands do not authorize network access.
Every remote operation needs a user request plus the exact confirmation
variables enforced by the script:

| Operation | Required confirmation |
| --- | --- |
| Remote read/status/output | `KAGGLE_REMOTE_CONFIRMED=1` |
| Kernel write/push | `KAGGLE_PUSH_CONFIRMED=1` |
| Dataset publication | `KAGGLE_DATASET_WRITE_CONFIRMED=1` |
| Source-attached run | `KAGGLE_FULL_DATASET_CONFIRMED=1` |
| Pull that may overwrite local files | `KAGGLE_PULL_CONFIRMED=1` plus remote confirmation |
| Workstream-specific run | Its additional spec-specific guard |

Never infer authorization from an old command, receipt, comment or completed
run. A consumed one-shot guard is not reusable.

## Identity And Provenance

- Authenticate with the user's own Kaggle account; never accept another
  person's API key.
- The authenticated username is the actor and owner only of resources created
  by that actor.
- Preserve every input and output using the exact canonical
  `owner/slug/version` returned by Kaggle.
- Different resources in one run may have different owners.
- Bind source, config, checkpoint and output hashes in local receipts under
  `runs/local/kaggle_launches` or the workstream's scored package.
- Keep credentials local, ignored and out of logs.

## Kernel Contract

Before any authorized push:

1. Read the active spec and `CURRENT.md`.
2. Resolve the exact kernel directory and immutable source versions.
3. Run the workstream-specific builder/preflight.
4. Run `validate` and `check`.
5. Confirm generated source is below Kaggle's script-size limit.
6. Confirm metadata privacy, accelerator, internet and source settings.
7. Confirm the worktree/provenance policy enforced by that guard.
8. Ask for explicit authorization naming the remote mutation.

Large checkpoints, manifests and binaries belong in versioned private input
datasets, not embedded base64 strings. Generated wrappers must validate their
payload manifest before importing project code.

## Generic Remote Pattern

Use a receipt-bound action whenever one exists:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check <kernel-dir>
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push <kernel-dir>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-launch <launch-receipt.json>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-launch \
  <launch-receipt.json> <output-dir>
```

Source-attached writes add `KAGGLE_FULL_DATASET_CONFIRMED=1`. Dataset
publication adds `KAGGLE_DATASET_WRITE_CONFIRMED=1`. Use the more specific
workstream action when available; it filters downloads and validates exact
receipts.

Pulling kernel source can overwrite local files and requires both read and pull
confirmation:

```bash
KAGGLE_REMOTE_CONFIRMED=1 KAGGLE_PULL_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh pull-launch <launch-receipt.json> <kernel-dir>
```

## Waiting And Monitoring

Do not hold an agent turn through a long Kaggle run. After one status check
confirms `RUNNING`, give the user a concrete local time to prompt again. A
remote poll always requires `KAGGLE_REMOTE_CONFIRMED=1`.

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh wait \
  <kernel-id> <poll-seconds> <max-polls> <max-queued-seconds>
```

`wait` distinguishes running timeout, queued timeout and terminal states.
Prefer receipt-bound status/output actions and never repeatedly download the
same output.

## Dataset Rules

- Attach datasets through `kernel-metadata.json` using exact slugs.
- The unsupervised development source is
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
- That source contains train/validation shards only; sealed evaluation uses the
  separate Specs 0017–0047 contracts.
- Supplemental masks are non-exhaustive and black pixels are unannotated.
- Latent binaries remain remote and immutable; download only the compact
  metadata/log files allowed by the workstream action.
- Never rebuild a published immutable input in place. Create a new version and
  bind it with a new receipt when a separately authorized contract requires it.

Exact data semantics live in `docs/behavior_inventory_kaggle.md`.

## Fail-Closed Guard Anchors

These strings remain because `scripts/kaggle_kernel.sh` checks them before
specific legacy-capable routes. Their presence is not launch permission:

- `synthetic no-dataset setup smoke`;
- `runtime_selection_kernel_ready`;
- `selected_runtime_debug_gate_contract_ready`.

All experiment training, sealed MIL/tissue/reconstruction evaluation and dense
rotation runs are complete. Their specific commands, versions and scientific
contracts remain in the relevant specs and receipts; this workflow does not
duplicate run diaries.
