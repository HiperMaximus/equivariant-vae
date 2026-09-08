# Spec 0037: Tissue Fast-Path Calibration Probe

Status: locked / implementation-ready
Owner/workstream: tissue-only preflight for the Spec 0023 label-efficiency campaign
Last updated: 2026-09-04

## Purpose

Measure the fastest finite paired train-step bundle for the already selected
three-class high-purity tissue classifier before the five 30-epoch development
runs are launched. This is a bounded runtime/AMP calibration probe. It does
not measure convergence, run validation, select an architecture or access any
sealed-test location.

## Frozen Scope

- Classes are `tumor`, `stroma`, and `necrosis`; this is not a healthy-versus-
  diseased task.
- Input is only the existing paired `float32 mu[16,32,32]` tissue training
  view. The 5,671-per-class manifest is the sole logical input. No validation
  or test CSV, label, pointer, or payload may be staged or read.
- The two branches use byte-identical `TissueClassifier` initialization from
  seed 3407 and identical logical batches; their latent values are the only
  intended difference.
- The model is the fixed Spec 0023 spatial encoder
  `16->32(k5,s2)->64(k3,s2)->128(k3,s2)`, GroupNorm(8)+GELU after every
  convolution, adaptive average pooling and `Linear(128,3)`. No dropout,
  augmentation, VAE fine-tuning, MIL pooling, attention, DDP, or test access.

## Batch Geometry

The full training manifests stay unchanged. `drop_last=True` is retained for
static compiled shapes, but the campaign batch is fixed by label budget:

| Per-class rows | Total rows | Batch | Steps/epoch | Tail |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 750 | 125 | 6 | 0 |
| 500 | 1,500 | 125 | 12 | 0 |
| 1,000 | 3,000 | 125 | 24 | 0 |
| 2,500 | 7,500 | 125 | 60 | 0 |
| 5,671 | 17,013 | 159 | 107 | 0 |

Batch 159 is the exact divisor of the full balanced pool; batch 125 exactly
divides every smaller nested pool. The former batch-128 confirmation is useful
stability evidence, but does not validate these exact static shapes.

## Probe Contract

For both batch shapes, run three fixed, maximally class-balanced batches from
the 5,671-per-class training manifest. Batch 159 is exactly `53/53/53` on every
batch. Because 125 is indivisible by three, its three compositions rotate
`42/42/41`, `42/41/42`, and `41/42/42` (tumor/stroma/necrosis); aggregate use is
therefore exactly 125 examples of each class. Before each branch/candidate case,
restore the seed-3407 model and RNG state and create a fresh fused,
non-capturable AdamW optimizer. One ordinary `torch.amp.GradScaler("cuda")` per
branch/candidate starts with PyTorch defaults (`init_scale=65536`, growth factor
2, backoff factor 0.5, growth interval 2000) and survives only across that
case's fixed calibration batches. A nonfinite gradient causes ordinary scaler
backoff, zeroes gradients, and retries the same loaded batch; three consecutive
backoffs on one batch are terminal. Calibration optimizer state never enters
simulated training.

After calibration, restore model/RNG exactly, create a fresh empty optimizer,
retain only the calibrated scaler and execute two warmup plus ten settled
simulated training updates at peak LR `1.5e-3`. All loss, logits, parameters
and optimizer state must remain finite. This is a numerical/runtime check, not
a learning gate.

The timed candidate set uses the same two-GPU assignment as the later tissue
campaign (normal on `cuda:0`, SO(2) on `cuda:1`), pinned PyTorch
`2.14.0/cu130`, FP16 autocast, channels-last latent/model layout,
`cudnn.benchmark=True`, `cudnn.deterministic=False`,
`torch.use_deterministic_algorithms(False)`,
`torch.set_float32_matmul_precision("high")` when present, fused AdamW, and
`set_to_none=True`. T4 has no TF32 tensor cores; record TF32 as unsupported
rather than treating a no-op setting as a speed result. Candidate code must
feature-detect, then time:

1. eager FP16 reference;
2. `torch.compile` full-graph numerical closure in `max-autotune-no-cudagraphs`;
3. `torch.compile` full-graph numerical closure in `max-autotune`;
4. each available compiled mode with `torch._dynamo.config.compiled_autograd`
   enabled.

The compiled closure is model plus FP32 cross-entropy and its AOTAutograd
backward. Ordinary GradScaler and fused AdamW remain native so standard overflow
semantics, including calibration, are not reimplemented privately. Compile
time is a non-cost; rank candidates by the slower branch's settled end-to-end
optimizer-step wall time for that shape. A candidate must use no graph break,
report generated-kernel/graph telemetry and retain finite committed updates.
No bitwise or FP32-equality gate is imposed.

## Later Campaign Contract

The probe winner is a shared runtime bundle, not a representation-specific
tuning choice. The later five-run campaign uses unweighted FP32 three-class CE,
the selected default-scaler calibration on startup, AdamW betas `(0.9, 0.999)`,
epsilon `1e-8`, matrix-only `5e-3` weight decay, and the shared peak `1.5e-3`.
The completed runtime probe's `1e-4` decay is immutable historical evidence for
that probe only; after reviewing its outcome, the user explicitly selected the
same `5e-3` campaign decay as MIL.
For each budget with `S` steps/epoch, it uses `W=ceil(0.1*S)`, 30 epochs and
`N=30*S` committed updates:

```text
lr(k) = peak * k / W                                      for 1 <= k <= W
lr(k) = peak * (0.01 + 0.99*(1 + cos(pi*(k-W)/(N-W)))/2)  for W < k <= N
```

Full natural-distribution validation remains at every half epoch. Both
representations retain individual best checkpoints by macro-F1, then lower CE,
then earlier boundary, but paired execution stops only when both exhaust the
five-epoch / ten-check patience or reach the 30-epoch ceiling. The five
development runs require a new implementation and separate authorization; this
probe does not launch them.

## Packaging And Remote Boundary

Stage one immutable private input dataset containing only the source snapshot,
the 5,671-per-class training CSV, a derived train-only manifest audit, fixed
probe-index batches, physical-parts catalog and exact producer locators. The
kernel attaches that input plus the exact latent producer kernels; it records
all runtime fingerprints and publishes only a compact probe result/log. Kernel
preflight rejects any staged logical `validation` or `test` file before opening
a latent store.

Remote dataset publication, kernel launch, status reads and output download
require the guarded Kaggle workflow and user authorization. This spec authorizes
only local implementation; the user explicitly authorized one subsequent
private calibration-probe publication and launch on 2026-09-04.

## Acceptance Criteria

- Exact batch geometry, paired initialization/order, source identities and
  train-only logical access validate before GPU allocation.
- Each calibration retries only its own already-loaded batch, never advances a
  cursor on overflow, and restores a pristine model/RNG plus empty optimizer
  before timing.
- Every candidate has finite committed simulated updates, runtime/graph/kernel
  telemetry and settled per-branch timing. The winner is selected jointly.
- Tests cover tail geometry, test/validation rejection, scaler reset/retry,
  no calibration-state leak, candidate feature detection, package source
  binding and output allow-list. Run focused tests, lint/type checks, kernel
  validation, shell syntax, `git diff --check`, preflight and independent
  clean-context review before remote publication.

## Non-Goals

No validation metric, convergence test, checkpoint selection, tissue test
access, final campaign launch, paper claim, or MIL architecture/runtime change.

## Related Files

- `docs/specs/0023-matched-supervised-latent-evaluation.md`
- `docs/specs/0035-largest-class-weighted-amp-probe.md`
- `docs/specs/0036-local-global-mil-training.md`
- `src/eqvae/models/supervised.py`
- `src/eqvae/data/supervised_latents.py`
