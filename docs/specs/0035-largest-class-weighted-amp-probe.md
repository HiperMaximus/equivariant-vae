# Spec 0035: Largest-Per-Class Weighted AMP Probe

Status: completed / Kaggle version 2 accepted
Authorization: consumed for private input dataset version 1 and kernel version 2; no further remote write
Owner/workstream: pre-learning numerical gate for Spec 0026
Last updated: 2026-09-03

## Goal

Verify the exact class-balanced FP16 training path of the accepted Spec 0026
MIL architecture on the largest complete training WSI in each diagnosis before
classifier learning is authorized. This is a disposable numerical/capacity
probe, not learning, model selection, validation or sealed-test access.

## Frozen Inputs

Derive only from Spec 0025's development manifests. The 106 training WSIs give
inverse-frequency weights `106 / (5 * n_class)`:

| Diagnosis | Index | Training WSIs | Weight | Largest WSI | Patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| CC | 0 | 23 | 0.9217391304347826 | 51346 | 29150 |
| EC | 1 | 24 | 0.8833333333333333 | 45630 | 32595 |
| HGSC | 2 | 40 | 0.53 | 35239 | 18981 |
| LGSC | 3 | 11 | 1.9272727272727272 | 57162 | 24031 |
| MC | 4 | 8 | 2.65 | 65094 | 23090 |

The compact private input bundle contains only these five bag rows, their
127,847 development instance pointers, the required physical catalog rows and
an authenticated source snapshot. Preserve every producer by its canonical
owner-qualified locator. The authenticated Kaggle user owns only newly created
dataset/kernel artifacts. The builder therefore requires the authenticated
actor name and renders that actor into the new dataset contract, metadata and
kernel source reference; the eight upstream producer locators remain unchanged.

## Exact Numerical Path

- Use the unchanged 1,513,055-parameter Spec 0026 model, Spec 0033 whole-bag
  fixed-25 local-attention backend and the exact PyTorch 2.14.0/cu130 runtime
  accepted by Spec 0034.
- Keep model parameters, parameter gradients, AdamW state, logits,
  cross-entropy and class weights in FP32. CUDA-resident latent maps and
  eligible autocast operations use FP16.
- With batch size one, compute ordinary unweighted FP32 cross-entropy. Do not
  use `CrossEntropyLoss(weight=..., reduction="mean")`, whose single-sample
  mean normalization cancels the class weight.
- Backpropagate `scaler.scale(unweighted_ce)`, call `unscale_` exactly once,
  record raw-gradient finiteness/norms, then multiply every FP32 parameter
  gradient by the sample's scalar class weight. Record weighted-gradient
  finiteness/norms before any optimizer step. The CE term is the entire
  optimized objective, so this is the same per-step gradient as explicit
  `weight_y * CE` without placing `weight_y` in the FP16 backward path. Require
  a positive data-gradient norm and verify that the weighted norm equals the
  raw norm times `weight_y`; AdamW weight decay must not disguise a fully zero
  or underflowed learning signal.
- Use `AdamW(lr=2e-4, weight_decay=1e-4 for matrix weights, fused=True,
  capturable=True)` and no gradient clipping. A missing or nonfinite gradient,
  loss, logit, parameter or optimizer state fails closed. Inspect every model
  parameter and AdamW state tensor after each applied update.

## GradScaler And Retry Policy

Use one ordinary `torch.amp.GradScaler("cuda")` per branch without constructor
overrides. Under pinned PyTorch 2.14 this means `init_scale=65536`,
`growth_factor=2`, `backoff_factor=0.5` and `growth_interval=2000`; assert these
defaults at runtime. Preserve normal upward exploration after 2,000 consecutive
successful updates and normal backoff after overflow. Save/restore scaler state
in any later learning checkpoint.

An overflowed attempt commits no optimizer update and therefore does not weight
that WSI. Clear its gradients and immediately retry the same already-loaded WSI
at the same LR without advancing sampler, scheduler or committed-step counters.
Permit at most three backoffs for one WSI, then fail with branch, WSI, diagnosis,
scale and named-gradient diagnostics. OOM, compiler errors, missing gradients or
post-class-weight nonfinites are terminal rather than retryable.

Calibration is a separate pre-training phase. Snapshot the exact initial model
state and Python, NumPy, Torch CPU and all CUDA RNG states immediately before it.
Visit the five stress WSIs in descending patch-count order: EC 45630, CC 51346,
LGSC 57162, MC 65094, then HGSC 35239. Before each WSI, restore that same model
and RNG snapshot and create a fresh disposable AdamW optimizer. Reuse the one
branch scaler across all five WSIs so overflow backoffs accumulate; after the
first successful commit for a WSI, discard its optimizer and restore again for
the next case. Calibration optimizer updates never become training updates.

After all five calibration commits, restore the initial model and RNG snapshot
exactly, assert byte/value identity, and create a fresh real-training optimizer
whose state is empty and whose step is zero. Retain only the calibrated scaler.
The first real epoch then uses its ordinary shuffle: it neither reserves these
five WSIs as a prefix nor removes them from the epoch. Every WSI therefore still
contributes exactly one successful real-training update per epoch, subject to
same-WSI overflow retry. The scaler keeps its normal growth/backoff policy and
is not frozen at the calibration result.

## Execution And Acceptance

For both normal- and SO(2)-latent branches:

1. Run the separate five-WSI calibration exactly as specified above: identical
   restored model/RNG state and a fresh optimizer per WSI, but one scaler across
   the full descending-patch sequence.
2. Restore the model/RNG boundary and verify that the fresh training-simulation
   optimizer starts empty at step zero while the scaler exactly retains its
   post-calibration state. Run five disposable training-simulation updates with
   that one optimizer/scaler in a fixed seeded shuffle distinct from calibration.
   The contract is `random.Random(3501).shuffle` over the descending list, which
   yields LGSC 57162, EC 45630, MC 65094, HGSC 35239, then CC 51346. This fixed
   probe order tests the boundary reproducibly; later real training uses its
   ordinary epoch shuffle.
3. Record every attempt, retry, scale transition, raw and weighted gradient
   summary, weighted reported loss, commit status, graph count/breaks and peak
   CUDA memory. Also record pre/post-calibration model/RNG identity, empty fresh
   optimizer state and the exact scaler state transferred across the boundary.

Acceptance requires all ten intended updates per branch to commit exactly once,
all five diagnoses to appear once in both calibration and training-simulation
rows, and no calibration model, RNG or optimizer state to cross the boundary.
The training-simulation optimizer must be newly created and empty at that
boundary; the scaler state must be exactly the calibration output. Require
finite raw and weighted gradients for every parameter, no post-weight overflow,
no graph break or eager fallback, at least one and no more than three compiled
numerical graphs, positive generated-kernel evidence, exact expected phase order
and matched WSI/label/weight/graph identity across both latent branches, and at
least 512 MiB conservative memory headroom. Report the minimum successful commit
scale across both phases as a diagnostic. It is not a manually frozen future
scale and is not evidence that every later training state is overflow-free.

## Remote Boundary

The consumed authorization covered private dataset
`maximshtefan/eqvae-largest-class-weighted-amp-inputs/1` and accepted kernel
`maximshtefan/eqvae-largest-class-weighted-amp-probe/2`. Kernel version 1 is
invalid because Kaggle rejected two source attachments before they were shared.
Any further retry, learning run, validation evaluation or sealed-test access
requires fresh explicit authorization.

## Kaggle Evidence

Version 2 produced `accepted_weighted_amp=true` with exact paired branch
identity. In both normal and SO(2) branches, EC 45630 overflowed at scales
65,536 and 32,768, then committed on the third allowed attempt at 16,384. The
other four calibration updates and all five training-simulation updates per
branch committed on their first attempt at 16,384. Model/RNG restoration,
fresh-optimizer boundaries, FP32 weighted-gradient proportionality, finite
AdamW state and the complete scaler-state chain all passed.

Both branches used two numerical graphs, added no graph during training
simulation and had zero graph breaks. Generated-kernel counts were 420/421.
Peak reserved memory was 10,733,223,936 / 10,712,252,416 bytes with
4,776,067,072 / 4,797,038,592 bytes conservative headroom. The immutable result
and log are under `runs/kaggle/largest_class_weighted_amp_probe_v2`; their
SHA-256 values are `e55d3131ed0e99743a3959fcf4193cfbdccf2d12a8c36b1d73c3661a4d4b8d7b`
and `59d514012706d4925a36c44dfaf845bfe6cd7a69f649d1f8091f61b39938a47c`.

## Local Verification

The revised focused Spec 0035 suite passes 30 tests. Touched-file Ruff and
BasedPyright pass. Independent local and downloaded-evidence P0/P1 reviews have
no remaining findings.
