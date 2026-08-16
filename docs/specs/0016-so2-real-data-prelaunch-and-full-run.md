# Spec 0016: SO2 Real-Data Prelaunch And Full Run

Status: locked / sessions 1-3 verified through update 27000 / session 4 authorized and staged
Implementation readiness: the exact checkpoint-only session-4 transport passes local preflight; private dataset publication/verification and guarded push remain
Owner/workstream: matched continuous-`SO(2)` training
Last updated: 2026-08-16

## Purpose

Bring the fixed Spec 0014 `SO2VAE` to the same pre-long-run evidence level as
the completed normal VAE, then package the first fresh 10-epoch run. Use the
normal run's selected recipe, batch, data, loss, beta, learning-rate schedule,
debug/resume bounds, fixed-32 proof, checkpoint cadence, and fixed-25 protocol
as the first attempt. This is one experiment path, not reusable product code.

## Non-Goals

- No architecture, basis, support, field-layout, runtime-recipe, beta, loss, or
  learning-rate search.
- No attempt to make the one-off launchers generic or portable.
- No claim that batch 25 is valid until the real dual-T4 prelaunch run passes.
- No remote Kaggle write/read and no full training without explicit permission.
- No normal-checkpoint reuse: the SO2 run starts from fresh SO2 parameters.

## Inputs And Data Contract

- Model: registry kind `so2_vae_fixed`, exact `SO2VAE`, latent width 16,
  43 learned convolutions, 34 radial gates, and `1,180,035` parameters.
- Runtime first attempt: the exact selected Spec 0011 plan at batch 25 per T4,
  global batch 50. Reusing this model-neutral recipe is a parity hypothesis,
  not an SO2 performance-selection claim.
- Data: `maximusshtefan/patches-pre-shuffled-ubc-ocean`, unchanged train and
  validation shards, `uint8` CHW `3x256x256`, normalized to `[-1,1]` on device.
- Tiny proof: the canonical real fixed-32 train selector, repeated only by the
  existing full-batch sampler.
- Objective/corruption: unchanged Tellez HED plus Gaussian corruption and
  `L1 + 0.1*(1-SSIM) + beta*KL`, with beta target `0.01`.
- Full schedule: 10 epochs, 6000 successful updates per epoch under global
  batch 50, 600-update linear warmup to effective LR `1e-3`, cosine decay to
  `1e-5`, validation/checkpoint/fixed-25 evidence every 3000 updates.

## Workflow Contract

1. Extend only the shared runner's architecture-specific gate evidence. Normal
   `GatedScalarActivation` rows and all normal-run behavior remain unchanged.
2. Snapshot every SO2 gate's `f0_a/f0_b/f1_a/f1_b` at run start. At an artifact
   boundary, rank 0 performs one eager no-grad FP16 forward on one real
   validation image with hooks outside the compiled training graph.
3. Emit exactly 68 SO2 rows: each exact one of the 34 real module names paired
   once with `f0_scalar` and once with `f1_radial`. Reuse the executed
   `_RADIUS_EPS`, FP32 gate math, and input-dtype output cast.
4. A row passes only with finite activation/gate/RMS evidence, present finite
   positive `a` and `b` gradients from the actual compiled training step, and
   finite positive parameter motion relative to run start. Missing evidence
   fails; no structural placeholder is legal.
5. The one-off prelaunch kernel runs, in order, at batch 25/rank: real debug to
   update 4, resume to update 8, then independent fixed-32 training to update
   128. It generates and validates the canonical selector before training and
   stops on OOM, compile/runtime mismatch, skip/nonfinite data, resume failure,
   gate failure, or less than 5% smoothed L1/reconstruction improvement.
6. The last 20 tiny updates form one settled real-loader measurement window.
   Synchronize only in this bounded window and record, on both ranks, data-wait
   time, complete optimizer-step wall time, peak allocated/reserved memory and
   headroom, plus post-settlement graph-break/recompile counts. The verdict uses
   the slower-rank mean step time and the exact
   `floor(300000/50) * step_time` epoch projection. This is measurement of the
   parity coordinate, not a batch or recipe search.
7. Only a passing downloaded prelaunch artifact can authorize the first SO2
   full session. Local packaging alone means `remote_pass_ready=false`.
   The user must explicitly accept the measured epoch/session cost before the
   full push; no speculative numeric speed threshold is invented locally.
8. The prelaunch verdict records the source commit and SHA-256 identities of
   the selected runtime plan, complete `src/eqvae` payload, Spec 0016 configs,
   model producer, shared runner, both launcher templates/metadata, and full
   config. The full preflight/push guard recomputes and requires those exact
   identities for the first full package. A continuation may change only its
   full checkpoint-transport wrapper, full metadata, and their consequent
   manifest commit/template entries. The authoritative session-1 payload is the
   downloaded `embedded_payload/payload_manifest.json` from clean commit
   `4aaf614f2cdbf1bc628e13858eb6c4e08300266b`. Its `src/eqvae`, both config trees,
   fixed-selector input, prelaunch wrapper/metadata, `pyproject.toml`, and
   `uv.lock` entries must remain byte-identical.
9. The separate full kernel started fresh, targets update 60000, and writes the
   normal-equivalent checkpoints, metrics, validation, and fixed-25 evidence.
   Session 1 committed update 9000 with SHA-256
   `1f53fe16aecf6382bf450cd0ac2be5db9fe2bbe6405dfcaa2c196cb40bca8e7d`.
   Session 2 copies only
   `runs/kaggle/so2_selected_runtime_full_v1_session1/checkpoints/step_009000.pt`
   into private dataset `maximusshtefan/eqvae-so2-session1-step9000` and pins
   `/kaggle/input/eqvae-so2-session1-step9000/step_009000.pt`. The upload payload
   contains only that file plus `dataset-metadata.json`. Each later session uses
   the same exact downloaded-latest-checkpoint pattern, validates path/hash before
   GPU work, and resumes through the shared checkpoint-only continuation path. No
   normal checkpoint/dataset may be attached or loaded.
   Session 2 committed update 18000 with SHA-256
   `5911ad37a1ed3f8a92055e45717be496d18545426e56667e1989a3da9a525ec4`.
   Session 3 copies only
   `runs/kaggle/so2_selected_runtime_full_v2_session2/checkpoints/step_018000.pt`
   into private dataset `maximusshtefan/eqvae-so2-session2-step18000` and pins
   `/kaggle/input/eqvae-so2-session2-step18000/step_018000.pt`.
10. Resume must restore the exact SO2 model, all optimizer groups/state, GradScaler,
   Python/NumPy/Torch CPU/CUDA RNG, named `train_data` and `train_corruption`
   generators, and DDP sampler progress. Session 3 requires
   `optimizer_step == successful_optimizer_update_count == 18000`, skips the first
   18000 batches by sampler indices without rereading their payloads, and rebases
   post-resume stochastic streams by rank. LR and beta remain derived from absolute
   successful-update count 18000, never replay warmup, and never advance on an AMP
   skip.
11. Compilation/startup cost and exact reproducibility remain non-goals. The
   paid run retains the selected speed-first runtime and successful-update AMP
   skip semantics.

## Config Contract

- One SO2 training base overlays the normal shared contract only to replace the
  model identity/count/layout metadata.
- Debug: 8 successful updates, checkpoint at 4, one validation batch.
- Tiny: 128 successful updates, checkpoint at 64, 10-update warmup then
  constant LR, exactly 32 unique real patches, 5% minimum smoothed L1 and
  reconstruction improvement, zero AMP skips/nonfinite rows, 68 passing gates,
  and one final 20-update synchronized timing/memory/compile-stability window.
- Full: fresh start, 10 epochs, beta `0.01`, normal-equivalent LR schedule,
  full validation, checkpoint/fixed-25 every half epoch.
- Kernel metadata attaches only the real UBC dataset for the first session.
  Session 3 requires the exact ordered `dataset_sources` allowlist
  `['maximusshtefan/patches-pre-shuffled-ubc-ocean',
  'maximusshtefan/eqvae-so2-session2-step18000']` and empty competition, kernel,
  and model sources. Any extra dataset or baseline/normal checkpoint slug/path
  fails closed.

## Outputs And Acceptance Artifacts

- Architecture-specific runner telemetry and focused mutation tests.
- `configs/spec0016/` SO2 base/debug/tiny/full configs.
- One private `eqvae-so2-prelaunch` script-kernel package.
- One private fresh `eqvae-so2-selected-runtime-full` package, not remotely run
  by the initial local implementation slice, then one-off checkpoint-only
  continuation packages as required.
- Downloaded prelaunch output later contains debug/resume/tiny summaries,
  actual 68-row gate CSVs, metrics, checkpoint proof, and a compact overall
  verdict.

## Acceptance Criteria

1. Normal gate telemetry is unchanged and SO2 without activation evidence
   remains fail-closed.
2. Synthetic focused tests prove exact 68-row identity, real activation/dtype
   fields, positive gradient/update requirements, and failure on missing,
   zero, nonfinite, mispaired, or duplicate evidence.
3. Local dry-run/config tests prove SO2 model selection and prohibit normal
   checkpoint contamination.
4. Prelaunch and full generated wrappers match their tracked templates and
   metadata, attach only intended datasets, use latest Torch before project
   import, and remain explicit-permission guarded.
5. Focused tests, `./scripts/python_quality.sh`, repo/workspace preflights,
   `git diff --check`, and two fresh read-only adversarial implementation
   reviews pass.
6. Remote prelaunch passes on two T4 ranks at batch 25 before the full kernel
   may be pushed. A batch-25 OOM/failure is a measured blocker, not permission
   for an implicit sweep; choose one corrected coordinate separately.
7. Prelaunch evidence reports per-rank and slower-rank settled step time,
   data-wait fraction, exact projected epoch time, peak allocated/reserved VRAM
   and headroom, and zero post-settlement graph breaks/recompiles. Full launch
   remains blocked until the user explicitly accepts that measured cost.
8. The first-session full guard rejects a prelaunch pass whose
   source/config/runtime hashes differ from the full execution inputs. A
   continuation guard independently requires the byte-identical embedded
   session-1 manifest entries named above plus its exact SO2 checkpoint source
   path/dataset slug/mount path/hash and exact two-dataset allowlist.
9. Full-run readiness means the downloaded prelaunch proof passes and the
   fresh full package passes local launch guards. It does not mean the full run
   has already been started or completed.

## Tests And Verification Commands

```bash
.venv/bin/pytest -q tests/test_selected_runtime_runner.py tests/test_so2_prelaunch.py tests/test_so2_full_run.py
./scripts/kaggle_kernel.sh preflight-so2-prelaunch
./scripts/kaggle_kernel.sh preflight-so2-selected-runtime-full
./scripts/python_quality.sh
./scripts/agent_preflight.sh
../agent_preflight.sh
git diff --check
```

## Implementation Blockers

Session 1 is downloaded under ignored
`runs/kaggle/so2_selected_runtime_full_v1_session1`. Its update-9000 checkpoint,
proof hash, schema-v5 state, source/config/runtime identities, two-rank metric
prefix, half-epoch validation/fixed-25 evidence, and 68 gate rows are verified.
The exact checkpoint dataset and session-2 wrapper/metadata/guards pass local
preflight and the full repository quality gate. No trainer or checkpoint-format
change was required. Commit `475e215` is on GitHub. Private dataset ID `11656723`
contains only the exact verified checkpoint and reports `isPrivate=true`. The
first guarded push was rejected before launch for zero remaining GPU hours. After
quota refresh, the clean-HEAD preflight passed again and Kaggle kernel version 2
ran to terminal `CANCEL_ACKNOWLEDGED`. Its separate downloaded output validates:
resume began at 9000, complete boundaries are 12000/15000/18000, and cancellation
interrupted boundary 21000 before commit. The accepted update-18000 checkpoint
SHA-256 is
`5911ad37a1ed3f8a92055e45717be496d18545426e56667e1989a3da9a525ec4`.
The user explicitly authorized this exact checkpoint upload/private destination
and session-3 launch. The one-off session-3 transport passes local preflight,
focused archive/resume mutations, and the full repository quality gate. Commit,
private dataset publication/verification, and guarded launch remain. Commit
`c8ff951` is on GitHub; the clean package preflight passes. After exact explicit
authorization, private dataset ID `11665702` was created with only the verified
checkpoint and reports `isPrivate=true`. The package was rebuilt from clean
commit `d251175`; Kaggle kernel version 3 terminated
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` after complete boundaries
21000/24000/27000. Its separate output is downloaded under ignored
`runs/kaggle/so2_selected_runtime_full_v3_session3`. All partial-manifest hashes,
the 9000-successful-update/rank metric prefix, 12 validation rows, 54 fixed-25
rows, 68 passing gate rows, and immutable originals are verified. The accepted
update-27000 checkpoint SHA-256 is
`7adfea7850ee7ab620f0363ca4a8fe9e41fd67160feeaeae1f07ff291a0bf6ba`.
The session-4 wrapper, metadata, guard, and checkpoint-only staging directory now
pin private slug `maximusshtefan/eqvae-so2-session3-step27000`, filename
`step_027000.pt`, its exact hash, update/config/runtime identities, session-3
payload authority, and unchanged execution core. The 16 focused tests and local
package preflight pass. After those exact details were reported, the user
explicitly authorized continuing the run; private dataset publication,
verification, and guarded launch remain.

## Known Risks

- The SO2 model has 4.383x the normal model's dense learned-convolution MACs;
  batch 25 may OOM or have unacceptable epoch time despite batch-1 readiness.
- An eager artifact probe can lie if it recomputes different gate semantics;
  tests must pin `_RADIUS_EPS`, cast behavior, exact module identity, and actual
  gradients/motion.
- Starting the full kernel before remote prelaunch passes would turn the parity
  hypothesis into an unbounded expensive guess.

## Adversarial Checks

- Delete/zero one SO2 gate gradient or update and require failure.
- Fabricate, duplicate, rename, mispair, or drop one of the 68 rows.
- Change F1 epsilon/dtypes or run hooks inside the compiled hot path.
- Point either launcher at a normal model config, normal run name, or normal
  resume checkpoint/dataset.
- Forge top-level pass while a nested phase, rank, checkpoint, or gate fails.
- Treat batch-1 readiness as proof of batch-25 feasibility or speed.
- Forge a fast rank-0 aggregate while rank 1 is slower, waiting on data, near
  OOM, or recompiling after settlement.
- Change the model, runner, runtime plan, or any Spec 0016 config after a green
  prelaunch artifact and require the full guard to reject it.
- Attach a normal checkpoint, a second checkpoint dataset, a renamed checkpoint,
  or the correct SO2 filename with the wrong SHA-256.
- Restore weights but omit optimizer/scaler/RNG/generator/sampler state; restart
  LR/beta warmup; replay skipped patch payloads; or advance schedules on AMP skips.

## Measured Remote State

- The exact batch-25 prelaunch passed on two T4s. The slower-rank settled mean is
  `2960.2401 ms/update`, projecting `4.93373 h/epoch`; peak reserved memory is
  `9366 MiB` with zero settled graph breaks/recompiles.
- Full session 1 published complete commit boundaries 3000, 6000, and 9000.
  Kaggle's file-list endpoint returned an empty list after cancellation, but the
  output archive and UI contain the complete files; the archive download is the
  authoritative source for continuation.
- Full session 2 published complete boundaries 12000, 15000, and 18000.
- Full session 3 published complete boundaries 21000, 24000, and 27000. At
  update 27000, clean validation L1/SSIM are `0.06347796463`/`0.7006000355` and
  deterministic-denoising L1 is `0.06682794668`. Its decoded rotation-consistency
  error continued improving, while the normalized latent ratio remained near
  `2.10`; do not conflate those two measurements.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0008-canonical-fixed32-and-remote-debug-tiny-readiness.md`
- `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`
- `docs/specs/0014-fixed-f01-full-vae.md`
- `docs/specs/0015-fixed-so2-selected-runtime-readiness.md`
- `src/eqvae/training/selected_runtime_runner.py`
- `kaggle/kernels/so2_runtime_readiness/run_template.py`
