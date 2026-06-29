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
- The historical working FSQ training reference is `kaggle/train_runs`. It
  trained correctly and is the source for the broad FSQ-successor
  macro-architecture and Kaggle runtime-efficiency ideas, but the new baseline
  and equivariant model remove FSQ quantization/codebooks/rounding/discrete
  latents and sub-pixel/PixelShuffle upsampling because they do not mix well
  with continuous `SO(2)` equivariance.
- Before the first full Kaggle run, benchmark where FP16/AMP avoids catastrophic
  failures and is actually faster, whether `torch.compile` is stable enough to
  repay its startup cost, and whether branchless full-batch corruption or indexed
  masked-sample corruption gives better throughput. Treat the useful historical
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
  actions exist; the selected-runtime runner derives the 10-epoch schedule as
  125000 optimizer updates with half-epoch intervals of 6250 updates;
  validation, checkpoint retention, best/final artifacts, resume hardening,
  artifact manifests, and the strict full-run verifier are in place; and
  adversarial fixes have been applied. Do not launch or poll the remote full
  run without fresh explicit user approval of the exact dedicated full-kernel
  command.
