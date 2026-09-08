# Spec 0030: Local-Global MIL Full-Bag Capacity Probe

Status: executed / closed with negative capacity result
Implementation readiness: complete; every remote authority is consumed
Owner/workstream: full-foreground WSI diagnosis architecture
Last updated: 2026-09-02

## Purpose

Prove that the exact Spec 0026 model can execute two paired optimizer steps on
the largest complete training bag before any learning package is proposed. This
is a capacity and numerical-sanity probe, not model learning or evaluation.

## Authorization Boundary

All three 2026-09-02 authorities are consumed. The final shared-access attempt
created private `maximshtefan/eqvae-wsi45630-local-global-mil-capacity/2`,
attached every required cross-owner source and loaded both complete bags. It
then OOMed in local block 2 during warmup at 15,338,887,168 peak allocated bytes
on a 15,636,037,632-byte T4; no optimizer step completed. No retry, fallback,
training, test access, dataset publication or visibility change is authorized.
The consumed authorization receipt identifier is
`spec0030_local_global_capacity_shared_access_retry_authorized`; retaining this
identifier documents the immutable one-shot guard and does not grant a retry.

## Fixed Inputs

- Train WSI `45630`, diagnosis index `1`, all `32,595` foreground patches in
  numeric `(y,x)` order.
- Existing private dataset
  `maximusshtefan/eqvae-wsi45630-capacity-inputs`, immutable version `1`, with
  contract SHA-256
  `99bb4d2f60558aee9691b67be4867ffae434bc306581a000fd5d72a6befac660`.
- Exact producer kernels, all version `1`:
  `maximusshtefan/eqvae-ubc-ocean-latent-run-04`,
  `maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up`, and
  `maximusshtefan/eqvae-wsi45630-completion`.
- The canonical model source is `src/eqvae/models/local_global_mil.py`; the
  builder copies and hashes those exact bytes into the launch package.
- One T4 per frozen-VAE branch. Both complete bags remain untruncated.

## Execution Contract

- Seed `1701`; byte-identical but storage-independent paired model states.
- Exact Spec 0026 parameter count `1,513,055`, graph identity, radius-two local
  topology, explicit FP32 sparse local attention and query chunk `8,192`.
- AdamW, learning rate `2e-4`, Spec 0026 semantic decay groups, matrix decay
  `1e-4`, vector/bias/relative-bias decay `0`.
- FP16 autocast around the model, FP32 classifier/loss, GradScaler initial scale
  `32768` and growth interval `1,000,000`.
- One warmup and one measured step. For each step both branches complete
  forward, backward, unscale and finite-gradient checks before either optimizer
  may step. Every parameter must have a finite gradient; both optimizer states
  must be complete; the scale must remain `32768`.
- Use the direct complete-bag path only. There is no checkpointing,
  recomputation, patch subsampling, graph alteration, or fallback. OOM is an
  accepted negative result that blocks learning and reopens Spec 0026.
- Record runtime fingerprint, input identities, graph identity, initialization
  identity, losses, step times, stage allocations, peak allocated/reserved
  bytes, device totals, scale, gradient coverage, and optimizer-state coverage.
- Always write one compact JSON artifact, including phase and exception details
  on failure. Never save latent vectors, model weights, predictions, or test
  information.

## Portable Package Contract

The tracked generated kernel contains only `run.py` and
`kernel-metadata.json`. Because Kaggle script execution guarantees only its
declared `code_file`, `run.py` embeds the canonical model as readable Python and
the canonical JSON contract; no base64 or hidden payload is used. Its source
metadata may carry a placeholder owner, but the generic push path rewrites only
that top-level owner in an ephemeral snapshot. Dataset and producer locators remain unchanged.
The accepted canonical owner/slug/version is saved in the generic immutable
launch receipt. A separate immutable local claim is written immediately before
the network attempt and blocks every second attempt under any account.

## Acceptance Criteria

1. The builder validates the existing verified version-1 input receipt, exact
   input contract/pointer hashes, source owner/version list, canonical model
   hash, package allow-list, Python compilation and sub-1-MB script limit.
2. Focused tests exercise package drift, graph construction, paired atomic-step
   gating, output schema, account-portable metadata, and the one-shot shell
   guard without a remote call.
3. `./scripts/python_quality.sh`, both preflights and `git diff --check` pass.
4. Independent clean-context reviewers find no P0/P1 correctness, leakage,
   provenance, portability, or launch-safety blocker.
5. Only then may the one authorized guarded private push occur.
6. A positive capacity result requires two committed paired steps, finite loss
   and every gradient, unchanged scales, complete optimizer state, and recorded
   VRAM headroom on both T4s.

## Commands

Local build and verification:

```bash
.venv/bin/python scripts/build_wsi45630_local_global_capacity.py build
.venv/bin/python scripts/build_wsi45630_local_global_capacity.py validate
.venv/bin/python -m pytest -q tests/test_spec0030_capacity_package.py
./scripts/python_quality.sh
./scripts/agent_preflight.sh
git diff --check
```

The historical package is closed. Do not invoke its push route again.

## Downstream Gate

If the probe passes, a separate spec and explicit authorization are still
required to package the development-only full-coverage reader and train the
classifier. The validation protocol must finish before sealed test access.
