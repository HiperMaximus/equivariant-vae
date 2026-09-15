# Spec 0003: Kaggle CLI Execution Workflow

Status: implemented
Implementation readiness: active workflow
Owner/workstream: remote execution
Last updated: 2026-09-08

## Purpose

Keep repository code and specs authoritative while using Kaggle only as a
receipt-bound remote execution surface.

## Contract

- Use `scripts/kaggle_kernel.sh`; do not use notebook JSON as executable truth.
- Kernel source lives under `kaggle/kernels`; generated `run.py` and payloads
  remain ignored.
- Remote permission exists only in the user conversation. It is never encoded,
  persisted, validated or consumed by repository code.
- The launcher is generic; it has no workstream-specific branches or permission
  state.

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

## Local Verification

```bash
./scripts/kaggle_kernel.sh help
./scripts/agent_preflight.sh
bash -n scripts/kaggle_kernel.sh
git diff --check
```

Exact operational commands live in `docs/kaggle_cli_workflow.md`; `CURRENT.md`
owns remote state.
