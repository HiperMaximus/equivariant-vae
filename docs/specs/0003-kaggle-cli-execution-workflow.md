# Spec 0003: Kaggle CLI Execution Workflow

Status: implemented
Implementation readiness: active workflow; no experiment launch authorized
Owner/workstream: guarded remote execution
Last updated: 2026-09-08

## Purpose

Keep repository code and specs authoritative while using Kaggle only as a
receipt-bound remote execution surface.

## Contract

- Use `scripts/kaggle_kernel.sh`; do not use notebook JSON as executable truth.
- Kernel source lives under `kaggle/kernels`; generated `run.py` and payloads
  remain ignored.
- Local build/validate/check does not authorize network access.
- Remote reads require explicit permission and
  `KAGGLE_REMOTE_CONFIRMED=1`.
- Remote writes require explicit permission and
  `KAGGLE_PUSH_CONFIRMED=1`.
- Dataset publication additionally requires
  `KAGGLE_DATASET_WRITE_CONFIRMED=1`.
- Source-attached execution additionally requires
  `KAGGLE_FULL_DATASET_CONFIRMED=1`.
- Pull additionally requires `KAGGLE_PULL_CONFIRMED=1`.
- Workstream-specific guards are cumulative and fail closed.

## Identity And Receipts

- The authenticated username is the actor and owns only newly created
  resources.
- Inputs retain their exact canonical owners.
- Every accepted resource records `owner/slug/version`, privacy, source order,
  hashes and a local receipt.
- Status, output and pull should resolve from a launch receipt rather than an
  inferred slug.
- Credentials are never stored, printed or committed.

## Kernel Package

- Embed repo code through the generated single-file payload contract.
- Validate the payload manifest before importing `eqvae`.
- Keep large checkpoints, manifests and binary data in immutable versioned
  private input datasets.
- Verify script byte size, metadata, accelerator, internet policy and source
  identities before push.
- Download only declared durable outputs; never treat compiler scratch or
  generated input data as evidence.

## Monitoring

Long-running jobs are not polled continuously in one agent turn. After one
status read confirms `RUNNING`, record the job identity and give the user a
concrete local time to prompt again. `wait` must bound both queued and running
states and surface terminal errors rather than hiding them.

## Retained Guard Contracts

These phrases are consumed by `scripts/kaggle_kernel.sh`. They are compatibility
anchors, not permission to run completed experiments:

- `kaggle_setup_smoke_ready` identifies the synthetic setup-only import and
  artifact check.
- The synthetic binary timing pretest workflow becomes Kaggle-push-ready only
  when its spec/config/metadata guards and user confirmation all pass.
- `runtime_selection_kernel_ready` identifies the bounded runtime selector.
- `selected_runtime_debug_gate_contract_ready` identifies the debug/resume/
  tiny-overfit proof surface.

The current experiment has no active Kaggle job. Training, sealed evaluation
and dense-orbit one-shot authorities are consumed.

## Local Verification

```bash
./scripts/kaggle_kernel.sh help
./scripts/agent_preflight.sh
bash -n scripts/kaggle_kernel.sh
git diff --check
```

Exact operational commands and confirmation-variable combinations live in
`docs/kaggle_cli_workflow.md`; `CURRENT.md` owns remote state.
