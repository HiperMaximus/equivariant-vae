# Spec 0039 — Paired Tissue Label-Efficiency Training

## Purpose and scope

Run the actual supervised **development** training campaign for the frozen
three-class tissue-patch classifier.  It compares frozen normal-VAE and
continuous-SO(2)-VAE latent representations at five nested balanced labelled
budgets.  This is not a MIL run and it does not access the sealed tissue test
split.

Spec 0037 established the selected T4 train-step runtime and AMP lifecycle. It
is runtime evidence only; this spec owns learning, development validation,
checkpoints, and result serialization.

## Immutable data and split boundary

The private input dataset contains only:

- `physical_parts.csv` and all twelve paired physical catalog identities;
- `tissue/tissue_train_{0250,0500,1000,2500,5671}_per_class.csv`;
- four derived WSI-stratified, class-balanced validation files at
  `250/500/1000/1250` rows per class;
- the canonical Spec 0023 manifest audit, selected source snapshot, initial
  state, contract and derived training configuration.

It contains neither the complete 31,339-patch natural validation file nor
`tissue_test.csv`, or any test location, count, source code path, prediction
surface, or test loader.  The canonical 31,572-patch test split remains
sealed.  A later test-evaluation contract must separately lock its WSI-cluster
bootstrap seed, replicate count, and output schema.

All data use the frozen Spec 0023 logical members and six exact producer
locators, in catalog order:

1. `maximusshtefan/eqvae-ubc-ocean-latent-run-01`
2. `maximusshtefan/eqvae-ubc-ocean-latent-run-02`
3. `maximusshtefan/eqvae-ubc-ocean-latent-run-03`
4. `maximusshtefan/eqvae-ubc-ocean-latent-run-04`
5. `maximusshtefan/eqvae-ubc-ocean-latent-run-05`
6. `maximusshtefan/eqvae-ubc-ocean-cancer-latent-top-up`

The campaign must verify mounted binary size/header geometry and sidecar
checksum/contract through `SupervisedLatentStore` before any logical read.
Logical membership is never resampled.  Both branches receive identical
initial weights and identical per-epoch shuffled logical indices; physical
reads may be grouped by shard.

## Fixed model and campaign matrix

Use the frozen `TissueClassifier`: the Spec 0023 `16→32→64→128` spatial
encoder (conv/group-norm/GELU, adaptive average pool) followed by `128→3`,
with `tumor=0`, `stroma=1`, `necrosis=2`.  No dropout, class weighting, or
architecture search is allowed.

| Per-class train labels | Per-class validation labels | Train / validation rows | Batch | Steps / epoch | Warmup updates | Total updates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 250 | 750 / 750 | 125 | 6 | 1 | 180 |
| 500 | 500 | 1,500 / 1,500 | 125 | 12 | 2 | 360 |
| 1,000 | 1,000 | 3,000 / 3,000 | 125 | 24 | 3 | 720 |
| 2,500 | 1,250 | 7,500 / 3,750 | 125 | 60 | 6 | 1,800 |
| 5,671 | 1,250 | 17,013 / 3,750 | 159 | 107 | 11 | 3,210 |

Every train pool divides exactly by its batch size, so `drop_last=True` loses
no training rows.  For `B=125`, batches rotate the only possible integer class
compositions `(42,42,41)`, `(42,41,42)`, `(41,42,42)`; over every consecutive
three batches each class occurs 125 times.  `B=159` uses 53 examples/class.
The validation subsets are nested prefixes of one deterministic per-class,
WSI-stratified priority order from the held-out validation view.  Within each
class/WSI, rows are permuted once with `PCG64(SeedSequence([20260904,
tissue_class_index,wsi_id]))`; each positive WSI contributes one row before
the remaining rows grow in normalized within-WSI-rank order.  Thus all
eligible held-out WSIs appear in every selection subset.  Inference streams
the selected complete subset without a training loader or test loader; its
final inference batch may be shorter and is never compiled.

For each budget, create fresh paired models from seed `3407`, serialize the
CPU state once, and restore it separately for the normal and SO(2) branches.
The same state hash is required for all branches and budgets.

## Optimization and runtime

- 30 maximum epochs and 10 minimum fully completed epochs; unweighted
  `CrossEntropyLoss` is evaluated in FP32.
- AdamW with `lr=1.5e-3` peak, betas `(0.9,0.999)`, `eps=1e-8`, `fused=True`.
- Decoupled `weight_decay=5e-3` for every parameter with `ndim >= 2`; zero for
  biases and GroupNorm/vector parameters.
- `W=ceil(0.1*S)` updates linearly from zero to peak.  Through update
  `N=30*S`, cosine decay reaches `0.01 * peak`, without restart.
- CUDA FP16 autocast; channels-last tensors/models; `cudnn.benchmark=True`;
  nondeterministic algorithms allowed; matmul precision high.  T4 does not
  use TF32.
- Use the Spec 0037 winner exactly where available:
  `torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)` with
  compiled autograd enabled for the model + FP32-loss/backward closure.  The
  fused optimizer and `torch.amp.GradScaler("cuda")` remain native.
- The non-training parent installs and validates the pinned runtime, then starts
  one fresh Python worker per representation.  Before each worker imports
  PyTorch, the parent sets that worker's `CUDA_VISIBLE_DEVICES` to its assigned
  T4 and assigns separate Inductor/Triton cache roots.  Each worker therefore
  sees one `cuda:0`, owns its entire compile/train/validation lifecycle, and
  writes an independent branch summary. The parent waits for children and
  validates/merges their summaries only. No Python training threads, DDP,
  `torchrun`, process group, all-reduce, gradient, parameter, optimizer or
  scaler synchronization exists.
- Before each real branch/budget, calibrate the ordinary default scaler on
  three fixed static batches.  Each calibration attempt restores the base model
  and RNG and has a fresh optimizer.  On a nonfinite loss/gradient, halve the
  scaler and retry that same batch, up to three backoffs.  The real optimizer
  is fresh and receives no calibration state; only the settled scaler carries
  forward.  The campaign never sets an arbitrary fixed scale.

## Validation, selection, stopping and artifacts

Validate after `floor(S/2)` and `S` scheduled updates each epoch, then write
one atomic branch boundary record. A branch improves only if its macro-F1 is
higher, or macro-F1 ties and mean cross-entropy is lower; an exact tie retains
the earlier boundary.  Retain each branch's `best`, `latest`, and `final`
weights-only checkpoints plus a manifest containing state hashes and cursor.

Patience is ten validation checks (five epochs), but it is not armed until the
end-of-epoch validation for epoch 10: the epoch-10 midpoint cannot terminate a
branch. Each branch runs until it has exhausted this armed patience or epoch 30
completes; a completed worker exits while the other continues, and the parent
merges both eventual results. This is not DDP:
each branch has its own native
`GradScaler`, optimizer and model update. An overflow therefore uses ordinary
independent PyTorch scaler skip/backoff behavior, as in MIL. The controlled
comparison pairs only initialization, logical examples, schedule and validation
boundaries; branch stopping is intentionally independent. It does not
synchronize gradients, parameters, optimizer state or scalar decisions. The
runner records each scaler state at validation boundaries.
Persist atomically at every validation boundary:

- per-branch calibration, update and validation histories;
- metrics: macro-F1, balanced accuracy, accuracy, mean CE, per-class F1 and
  a 3×3 confusion matrix;
- selected-validation predictions with frozen logical identity, truth, logits,
  prediction, and CE;
- per-budget branch order/access transcript hashes and selection/patience
  status; and
- an aggregate five-budget label-efficiency table and the runtime/config
  fingerprint.

The top-level result is `complete` only when all five budget pairs complete
and all branch checkpoints/hashes validate. A numerical, source or identity
failure is terminal and records its phase without consuming the sealed test
split.

## Packaging, checks and authorized remote actions

`scripts/build_tissue_training.py` builds a non-overwritable actor-portable
input bundle at `runs/local/tissue_label_efficiency_training`, with byte
records and source snapshot. `scripts/kaggle_kernel.sh` exposes build,
validate, preflight, input publish/receipt verification, guarded launch and
receipt-bound output download commands.  An input upload or kernel launch
requires a fresh explicit user authorization plus the existing Kaggle
confirmation variables; local implementation and validation do not publish.
The v1 launch claim is deliberately created before the remote push: if the
push has a transient failure, do not retry automatically. Version 1 failed
before a training update because two Python threads overlapped PyTorch 2.14
compiled-autograd contexts. With fresh explicit authorization, the repaired
v2 launcher reuses the byte-verified private input v1 through a separate,
non-overwritable retry package and one-use retry claim. It requires both the
verified input-v1 receipt and the v1 launch receipt, preserving v1 as
non-evidence. Kaggle completed v2 as
`maximshtefan/eqvae-tissue-label-efficiency-training/2` in 514.315 seconds;
the retrieved 105-file output matches its download receipt. All ten
branch/budget runs completed independently and stopped by the specified
10-check patience rule. The user then explicitly amended the future protocol
to require ten fully completed epochs before patience may stop a branch; the
separately guarded v3 retry reuses the verified private input v1 and records
this execution-policy amendment in every output. Kaggle accepted and completed
v3 as `maximshtefan/eqvae-tissue-label-efficiency-training/3` in 448.9 seconds
on 2026-09-04. All ten independent runs completed at least ten full epochs;
their terminal outputs are receipt-verified. Its distinct claim is
`runs/local/tissue_label_efficiency_training_retry_v3_authority/launch_claim.json`
and its canonical launch receipt is
`runs/local/kaggle_launches/maximshtefan/eqvae-tissue-label-efficiency-training/v0003.json`.
Inspect the retrieved development result before making any further remote action.

Acceptance requires focused unit/package tests for all geometry, schedule,
weight-decay grouping, package allow-list/test exclusion, source snapshot,
rendered kernel binding, validation selection and independent patience; static
lint/format and the repository quality gate are recorded in `CURRENT.md`.
