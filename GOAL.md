# Repo Goal

This repository supports the SIPAIM 2026 paper and the experiments behind it.
The thesis repository is separate and will be updated later from stable paper
results.

## North Star

Compare two genuinely comparable histopathology patch representation learners:

1. A non-equivariant denoising VAE built only from operations that have a clear
   path to the steerable implementation.
2. A continuous `SO(2)`-steerable denoising VAE using a repo-owned,
   compile-compatible implementation, with `escnn` as a reference, and with
   matched data, schedule, latent target, training budget, and evaluation.

The comparison should answer whether continuous rotation-equivariant structure
improves reconstruction, latent behavior, robustness, or downstream usefulness
for histopathology patches.

## Active Source Of Truth

- Hard repo instructions:
  `AGENTS.md`
- Architecture transition plan:
  `docs/equivariant_vae_transition_plan.md`
- Current handoff/status:
  `CURRENT.md`
- Kaggle CLI execution workflow:
  `docs/kaggle_cli_workflow.md`
- Historical Kaggle behavior inventory:
  `docs/behavior_inventory_kaggle.md`
- Issue-derived requirements and deliverables:
  `docs/repo_goal_and_requirements.md`
- Issue image inventory:
  `docs/issue_image_inventory.md`
- Settled decisions:
  `docs/decisions/README.md`
- Agentic adversarial review workflow:
  `docs/agentic_review_workflow.md`
- Spec-driven development workflow:
  `docs/spec_driven_development.md` and `docs/specs/README.md`
- Strict Python quality workflow:
  `docs/specs/0002-strict-python-quality-gate.md`
- SIPAIM paper source:
  `paper/sipaim2026`
- Overleaf sync workflow:
  `docs/overleaf_sync_workflow.md`

## Current Paper Target

- Conference: SIPAIM 2026.
- Format: IEEE conference full paper.
- Local paper source: `paper/sipaim2026`.
- Overleaf project:
  https://www.overleaf.com/project/69c614433cbc9e46cf226d24
- Tracked advisor-facing PDF: `paper/sipaim2026/sipaim2026.pdf`.

Before pushing paper changes to Overleaf or GitHub, refresh the PDF with:

```bash
./scripts/sipaim_overleaf_sync.sh compile
```

## Do Not Lose These Requirements

- Report SSIM, MAE, MSE, and PSNR with mean, standard deviation, and sample
  count `n`.
- Produce boxplots for the reconstruction metrics.
- Produce a training/evaluation dashboard analogous to the issue screenshots.
- Keep a fixed 25-patch validation set for qualitative reconstructions and
  visual checks.
- Generate original/reconstructed folders or grids for those 25 fixed patches.
- Add rotated-input qualitative artifacts using fixed continuous angles.
- Add `paper/sipaim2026/figures/rotated_input_vs_latent_grid.*`, comparing
  ground truth, rotated-input reconstruction, transformed-latent reconstruction,
  and error maps for the same patch/angle set.
- Implement an EQ-VAE-style latent visualization: top principal components,
  latent maps, transformed latent maps, and error/difference maps.
- Validate equivariance of nonlinearities, normalization, upsampling, VAE
  sampling, and latent statistics before running the full experiment.
- For the first repo-owned `SO(2)` convolution, use Gaussian radial shells times
  real angular harmonics with zero center support for spatial angular
  frequencies `m > 0`; keep Bessel/Fourier-Bessel as a future fallback only.
- Do not hide reconstruction boundaries with a final `tanh`; use a
  zero-initialized raw RGB output head and explicit clamped projections only for
  image-domain metrics/artifacts.
- Track learned activation gate health before full training, including `a,b`
  ranges, saturation, gradients/updates, and input/output RMS, so gate
  parameters cannot silently kill channels.
- The historical working FSQ training reference is `kaggle/fsq_train_reference.py`
  (the verbatim `train.py` extract — read this plain `.py`, not the raw notebook JSON
  `kaggle/train_runs`; it is the MINIMUM operational floor to match and ideally beat).
  Judge that floor by settled throughput/time-per-epoch and stable shared reconstruction
  quality, not FSQ-specific loss or discrete-latent metrics. It trained correctly and is
  the source for the broad FSQ-successor
  macro-architecture and Kaggle runtime-efficiency ideas, but the new baseline
  and equivariant model remove FSQ quantization/codebooks/rounding/discrete
  latents and sub-pixel/PixelShuffle upsampling because they do not mix well
  with continuous `SO(2)` equivariance.
  The FSQ run did not search its flags/hyperparameters: it proves that one complete recipe
  trained, not that batch 60, LR 5e-4, its DDP/compile pair, layout, optimizer, or loader
  choices were optimal. Use it and the immutable Spec 0011 v2 rows as strong priors for a
  compact search of reviewed complete bundles. This is a two-architecture tuning campaign,
  not an exhaustive inventory of every internal runtime value. Neutral or redundant options
  may remain in the winning recipe.
- Before the first full Kaggle run, choose the fastest settled TIME-PER-EPOCH recipe on the
  latest PyTorch release for the real dual-T4. FP16 AMP + `torch.compile` is the primary
  candidate, not a risky afterthought. Compile time is a NON-COST for a ~30h run, so
  `max-autotune` and relevant experimental/beta features are in-bounds. Measure a reviewed
  set of broad compatible DDP/compile bundles, including the verbatim FSQ
  `compiled_autograd=True` + `optimize_ddp=False` pair and useful modern overlap controls
  exposed by the installed runtime. A stable DDP partition/graph break that starts all-reduce
  earlier may beat a break-free graph; zero graph breaks is diagnostic information, not a
  universal eligibility rule. The timed train step
  must avoid `.item()`/`.cpu()` host synchronization and keep telemetry on-device. Corruption is
  no longer a benchmark axis — it is a FIXED runtime property (the vectorized inline
  `InlineStainCorruptor`, RNG swap `5dde097`); the blake2b branchless-vs-indexed
  throughput question is settled. Treat the useful historical
  FSQ efficiency flags as measured candidates too: cuDNN
  benchmarking/non-deterministic kernel selection, channels-last layout, DDP
  `static_graph` and `gradient_as_bucket_view`, optimizer/zero-grad fast paths,
  and any TF32 or matmul precision knob available in the Kaggle runtime. The
  first expensive run is performance-first: bitwise determinism and small
  numerical drift are acceptable if the row is materially faster and catastrophic
  safety checks still pass. Runtime-selection v5 selected the current fallback
  AMP-conservative dual-T4 runtime; the compact v6 relaxed scalar-gate AMP
  follow-up ran, was slower, and kept v5. Spec 0008 selected-runtime
  debug/tiny v5 passed on Kaggle and strict downloaded-output verification
  passed locally.
- Compile/startup time is not part of runtime selection for the long experiment. Kaggle is
  the primary test surface for CUDA, Inductor, dual-T4 DDP, VRAM, H2D overlap, and real
  throughput; use direct bounded Kaggle measurements instead of CPU approximations, with
  explicit permission for each remote write.
- Always upgrade Kaggle to the newest available PyPI torch before importing project code;
  never select or reject a recipe from Kaggle's older preinstalled stack or stale issue
  reports. Record the full runtime fingerprint (torch/CUDA build, driver, GPU identity/capability,
  compiler/backend versions) with every measurement and rerun selection if it changes before
  training.
- Treat each full training run as a multi-session job under Kaggle's 8-hour limit. Atomically
  flush metrics and save model/optimizer/scaler/scheduler/progress at least every half epoch
  (or sooner when projected wall time requires), leaving time to publish/download the artifact.
  End-only checkpointing is forbidden; exact RNG continuation is not required.
- Before writing `benchmark/selected_runtime.json`, time real dual-T4 DDP
  train-step rows. The selection benchmark must prove two visible T4s,
  `world_size = 2`, `nproc_per_node = 2`, per-rank device assignment, linked
  dataloader/numerical/corruption/gate evidence, and global throughput
  projection; missing, failed, or skipped dual timing blocks runtime selection.
- The first full run also needs passing dataloader-throughput, paired numerical,
  selected-runtime debug, checkpoint/resume, tiny-overfit, and gate-health
  checks on the selected runtime.
- Spec 0006 local mechanics are implemented and locally verified:
  shared v5 selected-runtime plan parser/application, strict linked
  runtime-proof status/write-decision/rank/return-code plus tokenized
  `torchrun --standalone --nproc_per_node=2` validation, UBC-format synthetic
  train mechanics, selected `indexed_masked` train corruption, clean validation
  RNG isolation, integrated simulated AMP skip progress semantics, checkpoint
  schema v5 with progress consistency checks before restore, fixed-32 selector
  readiness boundaries, observed local FP32/AMP-off row telemetry, and
  structured local readiness artifacts consumed by push readiness. The full
  plan-applied proof fails locally for unexecuted dual-T4 CUDA AMP/DDP fields,
  as intended.
- Spec 0007 local runner implementation is complete and locally verified:
  `eqvae.cli.selected_runtime_train` supports synthetic UBC-format dry-runs and
  real `ubc-pre-shuffled` roots, consumes the shared v5 selected-runtime plan,
  applies the selected batch/corruption/dataloader/zero-grad policy, includes
  the AMP/GradScaler train step with FP32 objective islands, records tokenized
  `torchrun --standalone --nproc_per_node=2` launch and DDP rank/device proof,
  applies selected DDP static-graph/bucket-view flags exactly, writes artifacts
  only on rank 0 after per-rank metric/gate gathers, records selected-runtime
  AMP/CUDA/DDP checkpoint-state statuses when active, bounds AMP-skip retries,
  blocks readiness on any AMP skip, and writes schema-v5 checkpoint/resume,
  train-step, gate-health, readiness, and artifact-manifest outputs. Keep this
  evidence non-promotable. Spec 0008 remote debug/tiny v5 is complete and is
  not approval for the long full run.
- Spec 0009 is implemented and locally verified for the first full
  selected-runtime run: a dedicated full-run kernel and guarded Kaggle script
  actions exist; under the new global-50 winner the selected-runtime runner derives the
  10-epoch schedule as 60000 optimizer updates with half-epoch intervals of 3000 updates;
  validation, checkpoint retention, best/final artifacts, resume hardening,
  artifact manifests, and the strict full-run verifier are in place; and
  adversarial fixes have been applied. Do not launch or poll the remote full
  run without fresh explicit user approval of the exact dedicated full-kernel
  command.
- Current status pointer (2026-08-13): active Spec 0011 is a lean, one-off Kaggle
  configuration search and full-run contract, not the superseded v4 audit platform.
  Direct dual-T4 probes
  selected per-rank batch 25 (global 50), FP16 channels-last compiled whole-step,
  Python DDP reducer/compiled autograd, fused AdamW, and the other settings recorded in
  `configs/spec0001/non_eq_vae_runtime_winner.json`. Two fresh measurements project
  about 4499 and 4672 seconds per 300,000-patch epoch. A compact hash-linked consumer
  plan, bounded 192-update LR-range v1, and strict 128-update fixed-32 overfit gate v6
  now pass on real dual-T4 Kaggle. Fixed-32 smoothed L1 improved 28.1% and reconstruction
  loss 22.5%. A clean beta-zero/deterministic/no-corruption probe improved L1 61.5%
  through 512 updates and remained descending, proving the network learns. The
  1024-update clean Kaggle
  v9 improved deterministic clean fixed-training L1 68.9% to `0.0788`, with every
  64-step bin descending. Its saved image is held-out validation and is not fixed-set
  memorization evidence. Paired v10 probes then showed beta `0.01` retained materially
  better deterministic reconstruction than beta `0.1` (L1 `0.07748` versus `0.09708`;
  SSIM `0.58764` versus `0.49837`) while keeping nonzero KL. The user locked beta `0.01`
  for the matched baseline/continuous-`SO(2)` comparison; do not run another beta probe.
  The normal-VAE baseline completed 60000 update counters across three checkpoint-only
  sessions and is locally verified with the user-approved single physical-update legacy
  exception. Clean/denoising L1 reached `0.05925/0.06236`, fixed-25 images and rotation
  artifacts are complete, and the final checkpoint is loadable. Specs 0012-0014
  now lock, implement, and locally verify the fixed 43-convolution continuous-
  `SO(2)` VAE at exactly `1,180,035` parameters. The next experiment gate is a
  separately authorized readiness slice: register this singular model in the
  selected runtime, add F0/F1 gate telemetry, and use one narrow direct dual-T4
  check for compile settlement, VRAM, and execution compatibility. Preserve beta,
  data, schedule, fixed examples, metrics, and downstream probes; do not reopen
  architecture/mechanics, recreate a tuner, or start full training without
  separate explicit authorization.
  This GOAL states the north star, not the frontier; read
  `CURRENT.md` and
  `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`.
