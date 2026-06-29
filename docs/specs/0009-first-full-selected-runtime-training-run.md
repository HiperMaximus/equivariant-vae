# Spec 0009: First Full Selected-Runtime Training Run

Status: full kernel version 1 pushed; awaiting approved status/output checks
Implementation readiness: guarded full-run workflow exists locally and passed
the required local verification gates; the first remote push was explicitly
approved and accepted by Kaggle, while status/output checks remain approval
gated.
Owner/workstream: comparable non-equivariant VAE baseline, first full Kaggle run
Last updated: 2026-06-29

## Purpose

Prepare the first full real selected-runtime training run after Spec 0008 proved
the narrow remote debug/resume/artifact/gate-health/tiny-overfit gate on Kaggle.

Spec 0008 proved that the selected v5 runtime can execute the real UBC data
surface in bounded debug and tiny-overfit modes. It did not create or approve a
long-run launcher. This spec owns the missing full-run contract, implementation,
preflight, and approval request for the first 10-epoch non-equivariant VAE
baseline training run.

The local implementation now adds the full-run config, runner mode, dedicated
Kaggle kernel, shell guards, local preflight, and strict full-output verifier
described here. Local verification and adversarial review passed. The first
remote full-kernel push was then explicitly approved and accepted as
`maximusshtefan/eqvae-selected-runtime-full` version 1 at
https://www.kaggle.com/code/maximusshtefan/eqvae-selected-runtime-full. Do not
run status reads or output downloads without fresh explicit approval.

## Non-Goals

- No Kaggle push, status read, or output download without fresh explicit user
  approval and the required confirmation variables.
- No reuse of `kaggle/kernels/selected_runtime_debug` as a long-run launcher.
- No runtime re-selection or relaxed-AMP experiment.
- No continuous `SO(2)` model implementation.
- No final paper result claims, metric tables, or issue closure from the first
  training run alone.
- No sealed masked-WSI test evaluation; that remains a later paper-claim gate.

## Inputs And Data Contract

- Selected runtime:
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
- Required remote proof prerequisite:
  `runs/kaggle/selected_runtime_debug_v5`, verified by
  `eqvae.cli.selected_runtime_gate --verify-output`.
- Full-run training config to add:
  `configs/spec0001/non_eq_vae_selected_runtime_full.json`.
- Base model/objective/corruption config:
  `configs/spec0001/non_eq_vae_model_base.json`.
- Real Kaggle dataset:
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`.
- Data mode:
  `ubc-pre-shuffled`, with `data_root = "auto"` on Kaggle.
- Train shard:
  `dataset/ubc_train_shuffled.bin` and
  `dataset/ubc_train_shuffled.csv`, 300000 patches.
- Validation shard:
  `dataset/ubc_ocean_valid.bin` and `dataset/ubc_ocean_valid.csv`,
  30000 patches.
- Patch shape:
  `3x256x256`, CHW `uint8`, normalized through the existing repo data path.
- Runtime topology:
  dual T4 DDP, `world_size = 2`, `nproc_per_node = 2`,
  per-device batch size `12`, global batch size `24`.
- Precision/corruption/runtime policy:
  v5 `amp_conservative`, FP16 autocast, GradScaler init scale `16384.0`,
  FP32 objective islands, no compile, selected `indexed_masked` train
  corruption, contiguous memory format, selected dataloader settings, and
  `zero_grad_set_to_none = true`.

## Outputs And Acceptance Artifacts

- Code artifacts:
  - full-run config;
  - full-run mode in `eqvae.cli.selected_runtime_train` or a narrow wrapper
    that delegates to it;
  - dedicated Kaggle kernel directory, for example
    `kaggle/kernels/selected_runtime_full`;
  - `scripts/kaggle_kernel.sh` actions for full-run preflight, push, status,
    and output download;
  - full-run output verifier.
- Remote artifacts after an approved run:
  - `benchmark/training_summary.json`;
  - `benchmark/selected_runtime_full_summary.json`;
  - `benchmark/selected_runtime_plan_applied.json`;
  - `benchmark/checkpoint_resume_proof.json`;
  - `benchmark/gate_health_summary.json`;
  - `benchmark/artifact_manifest.json`;
  - `metrics/train_steps.csv`;
  - `metrics/validation_metrics.csv`;
  - `metrics/gate_health.csv`;
  - checkpoints: final, best validation, and retained interval checkpoints;
  - nonblank reconstruction sample artifacts sufficient for handoff inspection.
- This first run may produce training-dashboard ingredients, but final paper
  figures/tables remain later evaluation work.

## Related Requirements And Evidence

- `GOAL.md`: first full run follows selected-runtime debug/resume/tiny proof.
- `docs/specs/0008-canonical-fixed32-and-remote-debug-tiny-readiness.md`:
  remote debug/tiny v5 passed.
- `docs/specs/0007-real-ubc-ddp-amp-selected-runtime-runner.md`: existing
  step-runner foundation.
- `docs/specs/0003-kaggle-cli-execution-workflow.md`: remote guard workflow.
- `docs/kaggle_cli_workflow.md`: Kaggle permission and polling policy.
- `docs/repo_goal_and_requirements.md` and
  `docs/issue_image_inventory.md`: later paper/evaluation artifacts.

## Architecture Or Workflow Contract

1. Add a separate full-run kernel.
   The selected-runtime debug kernel remains a bounded proof gate. The full run
   needs its own metadata id, ready marker, generated single-file wrapper,
   output directory default, guard, preflight, status, and output actions.
2. Derive full-run steps explicitly.
   The v5 selected runtime records `optimizer_updates_per_epoch = 12500`.
   The first run is 10 epochs, so the full-run target is exactly `125000`
   successful optimizer updates. Half-epoch boundaries are every `6250`
   successful optimizer updates.
3. Make epoch progress artifact-derived.
   The runner must not silently fall back to one step when given the baseline
   config. It must record requested epochs, updates per epoch, target updates,
   half-epoch interval, current epoch fraction, and resumed progress.
4. Full training must use stochastic VAE reparameterization.
   The full-run train path must sample `eps` from the seeded train generator, or
   call the model forward path that samples from that generator, so
   `z = mu + exp(0.5 * logvar) * eps` is stochastic during optimization.
   The existing zero/fixed-epsilon behavior is allowed only for deterministic
   debug/tiny proofs, paired numerical checks, validation views, reconstruction
   artifacts, and other explicitly deterministic evidence lanes. A full-run
   launcher must fail closed if it would train all optimization steps with
   zero epsilon.
5. Scheduled validation is required.
   At each half-epoch boundary, run a bounded validation pass that is explicit
   in the config. The first implementation should use `20` validation batches
   per interval for each required validation view: clean autoencoding and
   deterministic denoising. Final full-dataset validation and sealed test
   evaluation are later evaluator work.
6. Checkpoint cadence and retention must match the config.
   Save interval checkpoints at every half-epoch boundary, write `final.pt`,
   write `best_model.pt` from the best validation checkpoint rather than the
   final train row, and retain final, best, and the latest four interval
   checkpoints unless Kaggle output constraints force a narrower explicitly
   documented policy.
7. Resume must be long-run safe.
   Resume must restore or prove restoration of model, optimizer, LR/beta
   progress, GradScaler state, Python RNG, explicit NumPy generator, Torch CPU
   RNG, CUDA RNG state, named Torch generators, selected-runtime identity, and
   DDP sampler/progress state. A resumed run must continue from the correct
   optimizer update and sampler offset, not restart the data stream from the
   beginning.
8. Full-run readiness is distinct from debug readiness.
   Debug/tiny artifacts stay non-promotable. Full-run artifacts may claim
   full-run execution only after the dedicated full-run verifier validates the
   long-run artifact set, zero AMP skips unless explicitly reviewed, zero
   nonfinite metric rows, selected-runtime plan application, validation rows,
   checkpoint retention, and gate health.
9. Approval request is exact.
   The approval request must name the exact command, kernel id, selected
   runtime, target update count, projected duration, output directory, expected
   polling cadence, and confirmation variables.
10. Do not wait in-turn for the long run.
   After push and one immediate status read, if Kaggle reports `RUNNING`, record
   the state in `CURRENT.md`, give the user a concrete local time to resume,
   and stop active waiting.

## Config Contract

Add `configs/spec0001/non_eq_vae_selected_runtime_full.json` with:

- `status = "selected_runtime_full_run_ready_after_local_preflight"` only after
  implementation and local checks pass;
- `run.mode = "kaggle_selected_runtime_full_train"`;
- `data.kind = "ubc-pre-shuffled"`;
- `runtime.selected_runtime_required = true`;
- `runtime.runtime_config =
  "runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json"`;
- `runtime.required_debug_gate_output =
  "runs/kaggle/selected_runtime_debug_v5"`;
- `training.epochs = 10`;
- `training.optimizer_updates_per_epoch = 12500`;
- `training.max_train_steps = 125000`;
- `training.half_epoch_interval_steps = 6250`;
- `training.validation_batches_per_view = 20`;
- `training.validation_views = ["clean", "deterministic_denoising"]`;
- `training.train_reparameterization = "stochastic_seeded"`;
- `training.deterministic_eps_allowed_for =
  ["debug", "tiny", "numerical_checks", "validation", "artifacts"]`;
- `training.save_every_steps = 6250`;
- `training.checkpoint_retention =
  "best_final_latest_four_interval"`;
- `training.resume_supported = true`.

## Acceptance Criteria

1. A local preflight proves the full-run kernel is generated from the current
   source and is not the debug/tiny kernel.
2. The full-run wrapper refuses to launch unless downloaded Spec 0008 v5
   artifacts pass strict `--verify-output`.
3. The full-run command targets `125000` successful optimizer updates and
   records half-epoch interval `6250`.
4. The runner does not default to `max_train_steps = 1` for full-run configs.
5. Full-run optimization samples stochastic VAE epsilon from the seeded train
   generator and records/proves that zero/fixed epsilon was not used for every
   train step. Zero/fixed epsilon remains allowed only for deterministic
   validation, artifact, debug/tiny, and paired-numerical lanes.
6. Validation metrics are written at every half-epoch boundary for clean and
   deterministic denoising views.
7. Checkpoints are written at half-epoch boundaries, final, and best validation;
   retention is enforced or a stricter output-size policy is explicit.
8. Resume restores GradScaler, CUDA RNG, and sampler/progress state before
   continuing.
9. Full-run artifacts include selected-runtime plan application, DDP rank/device
   proof, AMP/scaler proof, stochastic train reparameterization proof,
   checkpoint/resume proof, gate health, train metrics, validation metrics, and
   artifact manifest.
10. The full-run push guard requires `KAGGLE_PUSH_CONFIRMED=1` and
   `KAGGLE_FULL_DATASET_CONFIRMED=1`.
11. Remote reads/status/downloads require `KAGGLE_REMOTE_CONFIRMED=1`.
12. Adversarial subagent review finds no high-severity launch blocker before
   requesting remote approval.

## Tests And Verification Commands

Required before asking for remote approval:

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_runner.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_full_run.py -q
./scripts/kaggle_kernel.sh preflight-selected-runtime-runner
./scripts/kaggle_kernel.sh preflight-selected-runtime-full
PYTHONPATH=src .venv/bin/python -m eqvae.cli.selected_runtime_gate --verify-output \
  --output-dir runs/kaggle/selected_runtime_debug_v5 \
  --runtime-config runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json
./scripts/python_quality.sh
git diff --check
./scripts/agent_preflight.sh
cd /home/maximus/Documents/Tesis && ./agent_preflight.sh
```

Local verification on 2026-06-29 passed after adversarial fixes:

- `PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_runner.py -q`
  passed with `8 passed`.
- `PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_full_run.py -q`
  passed with `10 passed`.
- `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner` passed with
  `52 passed`.
- `./scripts/kaggle_kernel.sh preflight-selected-runtime-full` passed with
  `11 passed`.
- Strict debug v5 output verification passed for
  `runs/kaggle/selected_runtime_debug_v5` against the v5 selected runtime.
- `./scripts/python_quality.sh` passed with `242 passed` and basedpyright
  `0 errors, 0 warnings, 0 notes`.
- `git diff --check`, repo `./scripts/agent_preflight.sh`, and workspace
  `/home/maximus/Documents/Tesis/agent_preflight.sh` passed.
- `/tmp` pressure during verification was resolved by deleting only known local
  scratch directories once, then running heavy gates with ignored repo-local
  `runs/local_tmp/...` scratch and deleting each scratch directory after use.
  The local preflight/quality scripts now default to process-unique
  self-cleaning scratch under `runs/local_tmp/...`, and runner preflight output
  is no longer left under `runs/local_preflight`.
- The first full-kernel push was later approved and accepted by Kaggle as
  version 1. No status read, output download, or full-output verification has
  been run yet.

Adversarial review results:

- The first implementation review found and the implementation fixed two
  high-severity local blockers: incomplete full-output evidence could pass, and
  resume proof was too weak for a long run. The verifier now requires complete
  per-rank train coverage, latest-four interval checkpoint hashes, and explicit
  GradScaler/CUDA RNG restore-attempt/restored evidence; the full kernel has an
  explicit resume-checkpoint env hook.
- A second post-fix review found no remaining high-severity blocker, but
  reported a medium dirty-bypass scoping issue in the full push guard and a low
  mislabeled full-summary checkpoint field. Both are fixed locally: the dirty
  bypass is accepted only in explicit local preflight guard mode and rejected by
  the real push guard, and full summaries now list interval checkpoints
  separately from final/best while the manifest hashes all checkpoints.
- The final post-edit review found no high-severity blocker and one medium
  resume-output issue: a resumed full run could overwrite metrics with only
  post-resume rows and then fail strict full-output verification. That is fixed
  locally: resume now loads and merges pre-resume train/validation rows, carries
  retained interval checkpoint metadata plus prior best-validation state, and
  full-run resume fails closed when required prior train history is missing.

Remote sequence and current state:

```bash
# Already approved and accepted as Kaggle kernel version 1 on 2026-06-29:
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_full

# Still requires explicit approval:
KAGGLE_REMOTE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh status-selected-runtime-full

# Still requires explicit approval after the run completes:
KAGGLE_REMOTE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh output-selected-runtime-full runs/kaggle/selected_runtime_full_v1
```

There is no `push-selected-runtime-full` action; the full launch uses the
generic guarded `push` action with `kaggle/kernels/selected_runtime_full`.

## Remaining Blockers

- No Kaggle full-run status read or output download has been approved or
  attempted after the accepted version 1 push.
- The strict full-output verifier is implemented and covered by synthetic
  artifact tests, but it cannot verify a real full-run output until an approved
  Kaggle run is downloaded.
- The next remote approval request must still name the exact status command,
  kernel id, selected-runtime artifact, target update count, projected duration,
  output directory, polling cadence, and confirmation variables.
- Final paper metrics, full evaluator artifacts, and sealed masked-WSI test
  evidence remain later specs/work items.

## Known Risks

- Accidentally pushing the debug/tiny kernel as if it were the long run.
- Accidentally exporting local preflight-only environment variables into a real
  push shell; the current guard rejects the known dirty-bypass variable outside
  preflight mode.
- Treating debug/tiny non-promotable artifacts as full-run evidence.
- Losing resume correctness after a Kaggle interruption.
- Producing a checkpoint-heavy output that is awkward to download or inspect.
- Overstating paper readiness before later evaluator/test artifacts exist.

## Adversarial Checks

- Try to invoke the full-run wrapper with the debug kernel metadata or debug
  ready marker.
- Remove or corrupt `runs/kaggle/selected_runtime_debug_v5` and verify the guard
  blocks remote approval.
- Provide a config with `epochs = 10` but no derived `max_train_steps` and verify
  launch fails rather than running one step.
- Resume from an interval checkpoint and prove GradScaler scale, CUDA RNG, and
  sampler/progress state continue correctly.
- Drop validation metric rows, gate-health rows, or interval checkpoints and
  verify the output verifier fails.
- Try a single-visible-T4 or wrong-accelerator environment and verify the guard
  fails closed.

## Open Questions

None that block implementation. The first full training run uses bounded
half-epoch validation for training supervision/dashboard evidence; full
validation/test metrics and paper figures are later evaluator work.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0003-kaggle-cli-execution-workflow.md`
- `docs/specs/0007-real-ubc-ddp-amp-selected-runtime-runner.md`
- `docs/specs/0008-canonical-fixed32-and-remote-debug-tiny-readiness.md`
- `docs/kaggle_cli_workflow.md`
- `src/eqvae/cli/selected_runtime_train.py`
- `src/eqvae/training/selected_runtime_runner.py`
- `scripts/kaggle_kernel.sh`
- `kaggle/kernels/selected_runtime_debug/run_template.py`
- `kaggle/kernels/selected_runtime_full/run_template.py`
