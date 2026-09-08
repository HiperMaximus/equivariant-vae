# Spec 0036: Local-Global MIL Training

Status: complete through repaired selected-best revalidation version 7
Implementation readiness: preserve evidence; decide separately on sealed-test release
Owner/workstream: paired downstream evaluation of the frozen normal and continuous-SO(2) VAE representations
Last updated: 2026-09-05

Remote authorization: on 2026-09-04 the user explicitly authorized the clean
paired restart and version-5 continuation; on 2026-09-05 the user explicitly
authorized continued learning through private version 6 from version 5 and the
revalidation-only version 7. Those authorizations are consumed; another
publication or launch requires fresh approval.

## Purpose

Train the fixed Spec 0026 local-global MIL classifier on the complete Spec 0025
foreground bags for the normal- and SO(2)-VAE latent representations. Preserve a
controlled paired scientific comparison while allowing each branch to execute,
checkpoint, retry numerical overflow, stop and finalize independently.

This first campaign is exploratory because it uses one fixed 23-WSI validation
split and one initialization seed. It may establish whether the architecture
learns sufficiently to justify a separately authorized sealed-test release; it
does not establish seed robustness by itself.

## Non-Goals

- No architecture, attention-backend, LR, seed, fold or hyperparameter search.
- No patch sampling, bag truncation, gradient accumulation, activation
  checkpointing, DDP or cross-branch synchronization.
- No sealed-test logical metadata, labels, indexed feature reads or metrics.
  Required physical binaries mix splits; their presence is not test release.
- No VAE retraining and no changes to the frozen latent stores.
- No publication, paper or thesis claims from development metrics alone.

## Inputs And Data Contract

- The only logical input is the exact six-file Spec 0025 `development/` bundle:
  `dataset.json`, `physical_parts.csv`, and train/validation instance and bag
  CSVs. Its externally pinned contract SHA-256 is
  `6aa25a3f56db62b903d3451e154c47dc0039233f88179c7bdab03b7a48319fe0`.
- Never stage the enclosing logical-dataset directory or `sealed_test/`.
- The physical catalog preserves all 15 exact owner-qualified Kaggle producer
  locators and byte identities. The authenticated actor owns only newly created
  input, resume and kernel artifacts.
- Each example is one complete FP32 `mu[N,16,32,32]` WSI bag. The reader must not
  sample, truncate, pad across WSIs or change logical order.
- Fixed split: 106 train / 23 validation. Class order is
  `CC=0, EC=1, HGSC=2, LGSC=3, MC=4`; train counts are 23/24/40/11/8 and
  validation counts are 5/6/8/2/2.
- The two branches use identical WSI membership, epoch permutation and model
  initialization. Only latent payload values differ.
- Construction and epoch-order seed: `1701`. Bootstrap analysis seed: `3601`.
- No stochastic augmentation or dropout is used.

## Frozen Model And Runtime

- Use the unchanged 1,513,055-parameter Spec 0026 classifier: staged
  `16->64->128->192` CNN, two radius-two fixed-25-plus-null local blocks,
  width 192, six width-32 heads, pre-norm residual attention and width-256
  SwiGLU, CLS plus 16 REGs, one calibrated-sigmoid all-patch read, one CLS-only
  17-token softmax read and an FP32 five-logit head.
- Install Spec 0033's state-compatible `WholeBagFixed25Attention` in both local
  blocks before moving or compiling the model.
- Before importing Torch, install only exact `torch==2.14.0` from PyTorch's
  official cu130 index with pip's cache disabled, then require the accepted
  PyTorch 2.14.0/CUDA 13.0 performance fingerprint. Do not install unused
  `torchvision` or `torchaudio`, and do not float to a newer stack without a new
  bounded backend/AMP bakeoff. Use FP16 autocast,
  channels-last latents/model and standard PyTorch norms under AMP for the
  patch/local stream. Keep the sigmoid scores, weights and reduction, summary
  output projection, complete 17-token global residual/SwiGLU and CLS-only
  block, logits and weighted CE in FP32. The unnormalized sigmoid reduction
  must never be cast back to FP16. Use fused non-capturable AdamW and no CUDA
  graphs.
- Compile the model plus FP32 CE numerical closure with Inductor,
  `fullgraph=True`, `mode="max-autotune-no-cudagraphs"`, isolated recompiles and
  recompile limit 3 when those arguments exist. Only axis 0 of the latent bag
  and three graph tensors is eligible to vary. Reject compile-disabling
  environment overrides. Record graph, break and generated-kernel counts as
  performance telemetry at checkpoint and final boundaries; private compiler
  counters must not decide whether a scientifically valid update commits.
  GradScaler and optimizer remain outside the graph.
- Each branch owns one GPU and one process, CUDA context, model, optimizer,
  scaler, reader, compiler cache and output tree. The parent creates one CPU
  initialized state, makes it available byte-identically to both children,
  waits for both, and aggregates status only. A branch failure or early stop
  never interrupts, rolls back, gates or truncates the other branch.
- Place Inductor and Triton caches under temporary storage, never beneath
  `/kaggle/working`; compiler artifacts are not experiment outputs. A terminal
  training numerical record must name its branch and phase, WSI, diagnosis,
  bag size, pre-attempt committed update/cursor and scaler value.
- Bind Torch/CUDA/cuDNN, both T4 names and capabilities, and NVIDIA driver
  versions into the canonical runtime hash stored in every checkpoint. A later
  session must match that hash before restoring either branch.

## Optimization Contract

- AdamW peak LR `2e-4`, betas `(0.9,0.999)`, epsilon `1e-8`.
- Ordinary convolution and linear matrices use weight decay `5e-3`; CLS/REG
  tokens, local null keys, biases, normalization parameters, one-dimensional
  offsets and semantic relative-bias tables use zero decay through
  `local_global_mil_adamw_parameter_groups`.
- Batch size is one WSI. Compute ordinary unweighted FP32 cross-entropy, then
  multiply that scalar by the class weight in FP32. Do not use weighted mean CE.
- The class scalar is `106/(5*n_class)`: CC `0.9217391304347826`, EC
  `0.8833333333333333`, HGSC `0.53`, LGSC `1.9272727272727272`, MC `2.65`.
- Per attempt: zero gradients with `set_to_none=True`; compute unweighted FP32
  CE and its FP32 class-weighted scalar; call ordinary `scaler.scale(loss)`,
  backward, `scaler.step(optimizer)` and `scaler.update()`. Do not manually
  unscale, edit or scan gradients, and do not inspect private AMP state. The
  reported objective remains class scalar times unweighted CE. No clipping.
- When GradScaler backs off, accept its skipped-update contract, clear gradients
  and retry the same already-loaded WSI at the same LR. Advance no cursor,
  permutation, scheduler, epoch, patience or committed-step counter. Permit at
  most three consecutive backoffs for one WSI, then fail that branch with
  failure-only named-gradient diagnostics. A successful commit resets the
  consecutive count naturally for the next WSI. Nonfinite scalar loss, OOM or
  compilation failure is terminal.
- LR is indexed by committed update `k` only. Warm up for five complete epochs
  (`k=1..530`) from 10% of peak to peak, including both endpoints:
  `2e-4 * (0.1 + 0.9*(k-1)/529)`. Then decay without restarts for
  `k=531..15900` with
  `2e-4 * (0.01 + 0.99*(1 + cos(pi*(k-530)/(15900-530)))/2)`. Thus the
  first update uses `2e-5`, update 530 uses `2e-4`, and update 15,900 uses
  `2e-6`.

## GradScaler Calibration

Each branch uses one ordinary `torch.amp.GradScaler("cuda")` with pinned runtime
defaults: initial scale 65,536, growth factor 2, backoff factor 0.5 and growth
interval 2,000. Do not manually pin the accepted probe's final scale.

Before training, snapshot the exact initial model and Python, NumPy, Torch CPU
and CUDA RNG states. Visit EC 45630, CC 51346, LGSC 57162, MC 65094 and HGSC
35239 in that descending-patch order. Restore the identical initial model/RNG
before each case, create a fresh disposable AdamW optimizer, and retry overflow
under the normal scaler policy. Reuse only the scaler across the five cases.

After all five cases commit, restore the initial model/RNG exactly, create the
fresh real optimizer with empty state, and retain only the calibrated scaler.
The ordinary first-epoch permutation still includes all five calibration WSIs.
A valid authenticated resume restores its saved scaler and skips calibration.

## Training, Validation And Selection

- Maximum 150 epochs, 106 committed updates per complete epoch (15,900 total).
- One deterministic without-replacement permutation per epoch, shared by both
  processes. Independent retries/stopping do not change either permutation.
- Before either latent branch is opened, derive every train/validation graph
  from the shared logical coordinates and write its canonical identity to the
  run contract. Each child rebuilds from the same metadata before reading that
  WSI's latents and must match the pinned identity. Because the identity hashes
  the coordinate, neighbour-index, valid-mask and radial-code arrays, equality
  enforces representation-independent graph arrays across branches.
- Validate the complete 23-WSI development split after committed updates 53 and
  106 of every epoch. Validation is inference-only and uses ordinary unweighted
  mean CE.
- Primary selector: WSI macro-F1. Ties use lower unweighted validation CE, then
  the earlier half-epoch boundary.
- Best-checkpoint selection is active from the first validation. Early-stopping
  patience is disabled and reset through the completed epoch-50 boundary
  (`k <= 5300`). Beginning at `k=5353`, patience is 40 consecutive
  non-improving validation checks (20 epochs). Branches maintain patience and
  stop independently, only at committed half-epoch boundaries; without a later
  improvement, the earliest stop is the completed epoch-70 boundary
  (`k=7420`).
- Record macro-F1, balanced accuracy, accuracy, ordinary mean CE, per-class
  precision/recall/F1/support, confusion matrix and total WSI count. Use
  zero-division value 0 for absent predicted classes.
- Record per-WSI logits, prediction, truth, bag size and graph-degree summaries
  so errors and confidence can be audited against cardinality/geometry.
- After both branches finish, compute paired normal-minus-SO(2) differences on
  their predictions at their independently selected best checkpoints with
  10,000 diagnosis-stratified WSI bootstrap replicates, percentile 95% CIs and
  seed 3601. These intervals capture validation-WSI sampling uncertainty, not
  training-seed uncertainty.
- Reload each selected best checkpoint after training and rerun the full 23-WSI
  validation split. Write the hash-bound observed rows and comparison record
  before applying a verdict. Require exact WSI/truth/prediction and discrete
  metric reproduction, finite logits/losses, and a uniquely highest recorded
  primary macro-F1 for the selected boundary. Record the revalidated CE and its
  signed difference from the selection-time CE, but do not require bit-identical
  floating loss under the intentionally nondeterministic fast GPU runtime. If
  predictions differ or the selected primary metric is tied, fail closed before
  paired bootstrap aggregation; a tied selection would require evidence from
  every tied checkpoint rather than an arbitrary numerical tolerance.

## Checkpoints, Resume And Outputs

- Commit a branch-owned atomic checkpoint at every validation boundary and on
  finalization. Retain rolling latest, independently selected best and final.
- A resumable checkpoint contains model, optimizer, complete GradScaler state,
  committed-step/schedule position, epoch and within-epoch cursor, current order
  and order hash, Python/NumPy/Torch CPU/CUDA RNG, train/validation histories,
  best selector state, patience, configuration/input/source/runtime hashes and
  a development-file/physical-source access transcript.
- Resume only from a fully committed hash-manifested boundary. Reject partial,
  foreign, stale, cross-branch or contract-mismatched state. Resume continues
  the saved permutation/cursor exactly and performs no new calibration.
- Checkpoints contain only `weights_only=True`-loadable primitives and tensors;
  selector and RNG structures are decoded through strict schemas before state
  restoration. The resume builder requires the exact versioned launch receipt
  and complete output-download receipt, verifies every staged byte, and creates
  a unique actor/hash-owned private dataset whose reference and contract hash
  are embedded in its continuation kernel.
- A continuation may contain either or both resumable branches. A branch that
  failed before producing a valid boundary is carried only as receipt-bound
  terminal evidence, is not relaunched, and can never contribute fabricated
  paired metrics. This preserves the valid peer's independent continuation.
- Per branch outputs: `calibration.json`, train and validation metric tables,
  WSI prediction records, `checkpoints/latest`, `checkpoints/best`, final
  checkpoint/summary, and resume proof. Root outputs: frozen run contract,
  runtime record, paired bootstrap analysis and overall status.
- Kaggle currently permits 12-hour CPU/GPU sessions. Pause only at a half-epoch
  boundary when elapsed time plus 1.25 times the recent half-epoch estimate plus
  the 15-minute save margin reaches 12 hours. A later session may resume each
  unfinished branch independently from its own latest boundary.

## Packaging And Remote Boundary

- The local builder produces an immutable private actor-owned input dataset and
  a two-file Kaggle script-kernel package. It embeds a byte-verified source
  snapshot of the executable runtime tree, exactly the six development logical
  files, all exact producer
  locators/hashes, the initial-state identity and this spec hash. The separate
  continuation builder creates immutable receipt-authenticated resume datasets
  and matching two-file kernels without contacting Kaggle.
- Visualization-only `artifacts/rotation_orbits.py` and its CLI renderer are
  explicitly absent from the MIL runtime snapshot; neither is imported by the
  launcher or training dependency graph. Their exclusion prevents unrelated
  concurrent paper-figure work from changing the executable package identity.
- Kernel preflight requires exactly two usable T4 devices, exact mounted source
  identities and absence of sealed-test logical files from the staged private
  development bundle before either child starts. Mixed physical producer mounts
  are expected and remain reachable only through the development pointer allow-list.
- Remote writes, learning/validation execution, retries, resume-dataset
  publication and output retrieval require the existing guarded CLI workflow
  and fresh explicit user authorization. This spec authorizes local code only.
- Sealed-test release remains a later explicit decision after configurations,
  source/input identities, both best-checkpoint hashes and the complete
  development history are frozen. There is deliberately no automated metric
  threshold that silently releases test data.

## Acceptance Criteria

- Focused tests prove model identity, exact weighting order, default-scaler
  calibration/restoration, same-WSI overflow retry, committed-update scheduling,
  half-epoch validation, selection/ties, independent stopping, atomic resume
  equivalence, shared graph identities, best-checkpoint revalidation, metrics
  and bootstrap determinism.
- Package tests prove the six-file development allow-list, sealed-test absence,
  exact owner-qualified sources, actor-owned outputs, source/config hash binding,
  two isolated subprocesses and absence of DDP/cross-branch gating. Resume tests
  additionally prove exact external contract binding, safe checkpoint loading,
  path confinement, receipt authentication and one-branch recovery.
- The production Python quality gate passes. An independent clean-context review
  has no unresolved P0/P1 findings.
- Local completion creates no Kaggle resource and reads no sealed-test file.
- Remote acceptance, when separately authorized, requires both branches either
  to finish normally or emit an independently resumable/terminal status with
  complete evidence. Development metrics are not sealed-test results.

## Known Risks And Adversarial Checks

- The 23-WSI validation set has only two LGSC and two MC cases; macro-F1 and
  checkpoint selection are noisy.
- One seed cannot measure optimizer/initialization variability.
- Variable-size bags can trigger extra compilation specializations; rely on the
  configured full-graph compiler contract and record compiler counts as
  telemetry rather than reimplementing a private runtime gate.
- Two concurrent readers/compilers can contend for four CPUs and storage. Record
  load, transfer, compile and compute timing before adding prefetch complexity.
- Do not rebuild/reverify the Python graph every epoch. Cache authenticated CPU
  graphs by WSI. The selected runtime lazily caches all train/validation graph
  tensors on their branch GPU only after projecting at most 512 MiB of graph
  storage against the accepted 4,776,067,072-byte minimum headroom; reject the
  cache before latent reads if either bound is exceeded.
- Probe-only parameter clones, per-parameter `.item()` calls, synchronizations,
  `empty_cache()` calls, manual AMP finite scans and exhaustive AdamW-state
  scans do not belong in the hot loop. Compact diagnostics run only on
  failure/boundaries.
- A reviewer must try to inject `sealed_test/`, alter a producer owner/version,
  resume one branch from the other's checkpoint, advance after overflow, change
  the epoch order after retry, couple branch termination, and exploit a partial
  checkpoint directory.

## Remote Version 3 Evidence And Repair

- Canonical launch receipt:
  `runs/local/kaggle_launches/maximshtefan/eqvae-local-global-mil-training/v0003.json`.
  The user stopped the run after the normal branch failed and while SO(2) was
  still training; therefore version 3 is incomplete and cannot support a paired
  result.
- `normal_vae` failed in the training loop with `Branch produced a nonfinite
  scalar loss`. Its old terminal detail contains only `attempt=1` and
  `kind=scalar_loss`, so the exact WSI is not recoverable. Its authenticated
  latest/best checkpoint is update 1060, exactly 10 complete epochs, with
  checkpoint SHA-256
  `eb290f3da0c613d2dc69d231107903cd0cdd1c5bab1065c6e1ce137949dc55e9`
  and manifest SHA-256
  `8a673cbdafdbff23356363130fddbff05a451e6e4945201509f8a311f4a550ad`.
- `so2_vae` had no terminal summary because it was stopped while running. Its
  authenticated latest checkpoint is update 1431 (13 complete epochs plus one
  half epoch), SHA-256
  `baf43c32f577742a4e2ac730599883ec8e13fc6b7ad299b3e2b84246db67ad58`,
  manifest SHA-256
  `0f942533a5a6efaab9e6679d8647b649a4d4cc55fccbbc4944f007f5dfa37790`.
  Its best checkpoint is update 1166, SHA-256
  `15ba7af6537c915a727deb91af29f74bf22d546531175d53a7247dd646aee31d`,
  manifest SHA-256
  `eadc90dae1e8e0aceba36a3afcc45930d4f564f444fa6161623defb05b497c63`.
- The three checkpoint objects and their byte-identical canonical
  manifests/metadata are preserved under
  `runs/kaggle/ubc_ocean_mil_training_v3_evidence` and pass the strict local
  checkpoint loader. The API's broad output download was intentionally not
  retained because version 3 accidentally saved tens of thousands of compiler
  cache files under `/kaggle/working`; future caches use temporary storage.
- The strongest identified failure mechanism is the old cast of an unnormalized
  sigmoid attention sum back to FP16 before the summary output projection. The
  repaired state-compatible model keeps that reduction and every later global
  operation in FP32. A regression forces a finite 128,000-valued summary, above
  FP16's 65,504 maximum, and proves the value remains finite and FP32. The old
  implementation fails this regression. This is a justified repair, not a
  retrospective proof of the exact version-3 intermediate because that run did
  not record intermediate finiteness.
- Future terminal numerical records include branch, phase, WSI, diagnosis, bag
  size, committed update/cursor and scaler. The existing version-3 resume builder
  mounts the old source and must be rejected after this repair. A clean restart
  is the controlled paired protocol; any migration requires a separately locked
  source/checkpoint contract and fresh remote authorization.

## Tests And Verification Commands

Focused development checks:

```bash
.venv/bin/python -m pytest -q tests/test_spec0036_mil_training.py tests/test_spec0036_mil_package.py
.venv/bin/ruff check <touched-python-files>
.venv/bin/basedpyright <touched-python-files>
bash -n scripts/kaggle_kernel.sh
git diff --check
```

Final production gate:

```bash
./scripts/python_quality.sh
./scripts/agent_preflight.sh
```

## Terminal Result

Private version 5 restored both version-4 branches and paused cleanly at its
12-hour deadline: normal is at update 6466/epoch 61 and SO(2) at update
6572/epoch 62. The authenticated output lives at
`runs/kaggle/ubc_ocean_mil_training_v5`; all six best/final/latest pointers and
objects pass the strict loader. Exact receipts, hashes and interim development
metrics are recorded in `CURRENT.md`. The exact next resume package was built,
validated, published as private dataset
`maximshtefan/eqvae-local-global-mil-resume-90ce36e22e912d16/1`, downloaded
again and byte-verified. Private version 6 reached early stopping for both
branches at updates 7420/7791; all best/final/latest checkpoints validate.
Finalization then failed for both branches at `Selected best checkpoint
revalidation differs`: the runner compares the full metric dictionary with
exact float equality under the nondeterministic fast runtime and raises before
recording observed revalidation rows. Version-6 receipts, hashes and the bounded
diagnosis are in `CURRENT.md`. Preserve all checkpoints and do not continue
learning. The repaired finalizer writes evidence before verdict, applies the
tolerance-free categorical/unique-primary contract above, reuses stopped final
checkpoints and performs zero optimizer updates. Its package and private dataset
are byte-verified. Private version 7 completed: both selected-best
revalidations pass with exact WSI identity, predictions and discrete metrics;
all six checkpoint slots validate; and the paired bootstrap is authenticated.
Normal/SO(2) selected validation macro-F1 is `0.44808`/`0.59143`; the paired
normal-minus-SO(2) difference is `-0.14335` with 95% interval
`[-0.28685, 0.02167]`. Training-set weighted loss continued declining while
validation patience expired, so the stopping rule separated continued fit from
generalization. Exact artifact hashes and loss-trend evidence are in
`CURRENT.md`. Sealed-test release remains separately gated.

## Open Questions

- Whether this one-seed exploratory development campaign is sufficient to
  justify sealed-test release is a later human scientific decision, not an
  implementation default.
- Additional seeds or grouped cross-validation require a new precommitted spec
  and compute authorization; they are not implicit retries of this campaign.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0023-matched-supervised-latent-evaluation.md`
- `docs/specs/0025-full-foreground-logical-mil-dataset.md`
- `docs/specs/0026-local-global-sigmoid-mil.md`
- `docs/specs/0032-amp-fixed26-mil.md`
- `docs/specs/0033-exact-local-attention-backend-bakeoff.md`
- `docs/specs/0034-full-compiled-fixed25-mil-probe.md`
- `docs/specs/0035-largest-class-weighted-amp-probe.md`
- `docs/kaggle_cli_workflow.md`
