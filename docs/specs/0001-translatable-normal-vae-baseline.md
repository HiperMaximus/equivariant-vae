# Spec 0001: Translatable Normal VAE Baseline

Status: implemented, trained and frozen at update 60000
Implementation readiness: closed; no new training launch authorized
Owner/workstream: matched non-equivariant control
Last updated: 2026-09-08

## Purpose

Define the normal denoising VAE control whose operations, data, latent target
and evaluation are directly comparable with the continuous-`SO(2)` model.

## Frozen Contract

| Item | Value |
| --- | --- |
| Input | RGB `256x256`, normalized to `[-1,1]` |
| Posterior | Gaussian `mu, logvar`, shape `(B,16,32,32)` |
| Sampling | `mu + exp(0.5 * logvar) * eps` |
| Learned parameters | `3,958,435` |
| Training budget | 60,000 committed optimizer updates |
| Beta target | `0.01` |
| Downsampling | fixed anti-aliased blur plus decimation |
| Upsampling | bilinear resize plus convolution |
| Output | raw zero-initialized RGB convolution; no final `tanh` |

The architecture is ResNet-like and uses ordinary `Conv2d`, GroupNorm and
learned scalar gates. It excludes FSQ, codebooks, rounding, discrete latent
telemetry, PixelShuffle, nearest-neighbor upsampling and arbitrary pointwise
adapters that would make the matched steerable branch unfair.

## Data And Objective

- Development source:
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
- Train and validation WSI membership is disjoint from the 152 masked held-out
  WSIs.
- The corruptor produces the denoising input; the clean normalized patch is the
  target.
- Reconstruction loss uses raw output. SSIM, PSNR and images use an explicit
  clamped projection.
- KL, reconstruction, beta, posterior and output-range telemetry remain
  separate.
- Validation contains clean and deterministic-denoising views.

Corruption RNG policy is speed-first and does not require bit-identical
per-sample draws. Clean evaluation identities and statistical artifact inputs
remain deterministic where stable identity is the measured contract.

## Runtime And Training

Spec 0011 owns the selected dual-T4 compiled runtime. The runner:

- distinguishes physical attempts, AMP skips and committed updates;
- keeps step timing free of avoidable device-host synchronization;
- writes checkpoints and metrics atomically at verified boundaries;
- verifies DDP rank/device assignment, loader integrity, corruption, numerical
  behavior, gate health, checkpoint resume and tiny-overfit behavior;
- preserves the accepted single physical-update exception recorded in
  `CURRENT.md`.

Final checkpoint:
`runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`,
SHA-256 `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075`.

## Evaluation

- Spec 0010 owns the deterministic fixed-25 protocol.
- Spec 0044 owns advisor-requested validation metrics and plots.
- Spec 0045 owns the full sealed reconstruction test.
- Specs 0026, 0036, 0039, 0041 and 0043 own downstream supervised evidence.
- Spec 0048 owns final report interpretation.

No validation or test result permits retraining, tuning or checkpoint changes.

## Retained Fail-Closed Tokens

These tokens remain because `scripts/kaggle_kernel.sh` validates legacy-capable
routes. They do not authorize a remote write:

- `kaggle_smoke_ready`;
- `kaggle_synthetic_timing_contract_ready`;
- `real_data_runtime_pretest_contract_ready`;
- `v8_shortlist_eager_amp_then_dual_gate`;
- `selected_runtime_debug_gate_contract_ready`.

## Verification

Production changes touching this model or runner require focused model,
dataloader, loss, checkpoint and selected-runtime tests plus:

```bash
./scripts/python_quality.sh
./scripts/agent_preflight.sh
git diff --check
```

The model, runtime search and training campaign are closed. Preserve the frozen
checkpoint and evidence.
