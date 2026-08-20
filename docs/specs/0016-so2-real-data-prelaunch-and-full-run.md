# Spec 0016: SO2 Real-Data Prelaunch And Full Run

Status: locked / checkpoint lineage verified through update 54000 / session-4 67/68, session-5 66/68, and session-6 66/68 caveats retained / session-6 deterioration explicitly accepted / fresh session-7 continuation locally verified and awaiting remote identity plus launch authorization
Implementation readiness: session-5 version 1 used Kaggle's legacy input root; minimal probes proved the correct checkpoint mount is `/kaggle/input/datasets/maximshtefan/eqvae-so2-session4-step36000/step_036000.pt`. Earlier T4 requests returned CPU workers, but post-verification version 4 proves `torch 2.10.0+cu128`, CUDA `12.8`, and two Tesla T4 devices; version 5's no-data `pip --dry-run --upgrade torch torchvision torchaudio` returns zero and resolves CUDA-enabled Torch `2.13.0`. Commit `3d5bf766f323645d725f92a9dd5e27deaf438b7b` changes only that mount path and its wrapper hash; the clean generated package, 16 focused tests, and full `797 passed, 1 skipped` quality gate pass. After exact user authorization, Kaggle version 2 ran from clean GitHub commit `e1b9e9f9a28299f4604a768720345ae9cd7c2fb3` with only the public UBC and exact private step-36000 datasets. It reached terminal `CANCEL_ACKNOWLEDGED`; its freshly downloaded output proves the complete update-45000 boundary. The locally staged session-6 45k transport has the exact two-file dataset input, repinned wrapper/metadata/guard/tests, passing 16-test preflight, and final `797 passed, 1 skipped` quality gate. After exact authorization, GitHub `main` advanced to `a4f7a685d2a1deee6357fae1586f0c7973c20e5e`; private dataset `maximshtefan/eqvae-so2-session5-step45000` version 1 (ID `11701181`) verified `isPrivate=true` and only the 16,440,368-byte checkpoint; Kaggle kernel version 3 was submitted. Its terminal log proves Torch 2.13/CUDA installation completed, then it failed before checkpoint load at the correct namespaced 45k path because Kaggle omitted that input mount. A subsequent private CPU-only/no-internet probe attached only that dataset and verified the exact path, bytes, and SHA-256 `703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93`. With fresh exact authorization, unchanged version 4 was submitted through update 60000; its terminal log repeats the same missing-path failure after successful Torch/CUDA installation. The pulled v4 server script retains the correct private slug/path, and server metadata lists both source slugs while normalizing their order to private then public. The exact T4/Internet no-Torch/no-training two-source probe completed with both mounts and the exact 45k bytes/hash, ruling out the source pair itself. The exact preamble probe then found the file at exact size before the same Torch upgrade and at exact SHA-256 immediately after its 127-second completion, with zero wait; payload extraction can only write under `/kaggle/working`. These controls isolate intermittent Kaggle full-worker attachment behavior. Under fresh authorization, the transport now waits read-only for at most 600 seconds in 10-second logged increments before the unchanged byte-count/SHA-256 validation; timeout still fails before training. The wrapper SHA-256 `f9210ea3d8fc3b9739d74e0aef69821c4e5bd0af612edb5a6c62743fe91e262c`, 17-test continuation preflight, and full quality gate pass. Kaggle accepted it as session-6 kernel version 5 through update 60000; submission alone creates no training/output authority until its own later proof/manifest does.
Owner/workstream: matched continuous-`SO(2)` training
Last updated: 2026-08-20

Current terminal evidence: bounded-wait version 5 remained without the private
mount for all 600 seconds and failed before checkpoint load or training.
Server-accepted v5 and successful preamble-probe metadata match exactly on T4,
Internet, Docker image, and both dataset sources; only their kernel resources
differ (`131085532` fails, new `131241843` passes). The preamble probe used the
same literal checkpoint path/hash but not the full wrapper's checkpoint helper
calls, so it is not the final exact-call control. A disposable replacement is
locally staged from the generated full `run.py`; all source through line 8922
is byte-identical, and only its final entry point changes to stop after the
unchanged Torch-upgrade, payload-extraction, path-resolution, mount-wait, and
byte/hash-validation sequence. Its local authoritative-45k check passes. After
exact authorization, private exact-call probe version 1 completed; the
downloaded proof validates the exact Kaggle path, 16,440,368-byte size, and
SHA-256 `703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93`
without launching distributed training. The existing full kernel resource is
the proven provider-broken coordinate. The unchanged continuation is locally
repinned to fresh private kernel
`maximshtefan/eqvae-so2-selected-runtime-full-session6`; only ID/title and their
builder/guard/test pins changed. Wrapper SHA-256
`d66ba76f72d7fbbdc85dc991260c6ad55a4ef9f60fb683df31b8201d97e5af63`,
17 focused tests, and the full `798 passed, 1 skipped` quality gate with zero
type errors pass. After exact authorization, Kaggle accepted private kernel
version 1 from clean commit
`396d897dc442b5e5f9f94e32f01679d35fa69858`. It ended
`CANCEL_ACKNOWLEDGED` after the complete update-54000 boundary. The downloaded
proof names `checkpoints/step_054000.pt`, size 16,440,368 bytes, SHA-256
`2ae4785571e2d1b4e690957e3cf74f749c7e273f1701ee274cc7b2b2e4a8742c`,
with checkpoint/metric prefix 54000 and complete model, optimizer, scaler, RNG,
sampler, schedule, and beta state flags. All 13 manifest hashes verify. Four
synchronized AMP-recovery attempts produced eight rank rows and zero successful
nonfinite updates. Gate health remains 66/68: one saturated-open channel in
`decoder_blocks.2.output_gate:f1_radial` and five in
`encoder_blocks.6.main_gate:f1_radial`, both with finite positive gradient and
update evidence. Preserve that channel-count deterioration. On 2026-08-20 the
user explicitly accepted it after confirming training continued correctly.
The final local transport stages only `step_054000.pt` plus metadata for private
slug `maximshtefan/eqvae-so2-session6-step54000` and reuses the known-good
private kernel `maximshtefan/eqvae-so2-selected-runtime-full-session6`.
Wrapper SHA-256
`a66f8134b63169ec9cfe03d621f52141920d92feca9b80d2cca0271fae5c13fd`,
17 focused tests, and the full `798 passed, 1 skipped` quality gate with zero
type errors pass. After exact authorization, private dataset version 1 (ID
`11716939`) verified `ready`, `isPrivate=true`, the sole 16,440,368-byte
`step_054000.pt`, and the hash-pinned description. Kaggle accepted private
kernel version 2 through update 60000, but it ended `ERROR` before checkpoint
load: remote metadata records both exact dataset sources and the exact local
script hash, while the ready single-file checkpoint dataset never materialized
in the worker during the 600-second wait. Treat this as the same provider-side
private-input mount failure class, retain update 54000 as sole authority, and
require new exact authorization for any probe, fresh resource, or retry. The
staged minimal fresh-resource check at
`kaggle/kernels/so2_session6_step54000_mount_probe/` validates the exact path,
bytes, and hash with both production datasets, then stops without training;
the full `798 passed, 1 skipped` quality gate with zero type errors passes, and
after exact authorization Kaggle accepted private probe version 1. Its bounded
20-minute wait ended `TIMEOUT_STILL_PENDING`, then a later authorized check
found the same version `COMPLETE`. Its terminal log and downloaded JSON prove
the exact 16,440,368-byte private file and expected SHA-256 mounted on a fresh
resource configured with both production source slugs; it did not inspect UBC
files. Treat this only as fresh-resource checkpoint-mount evidence. Kaggle's
internal cause remains unproven, but a fresh resource is the best-supported
workaround. The staged private session-7 full resource preserves the exact 54k
checkpoint, source allowlist, T4/Internet settings, execution core, trainer,
runtime, schedule, 600-second fallback, and update-60000 target. It moves exact
mount/byte/hash validation before the Torch upgrade and changes only ID/title
plus matching transport pins. Wrapper SHA-256 is
`03887128886879b8c2ac4e68b210233dcbcbc181344c224a3990bb34824a8dd0`;
the focused 17-test continuation preflight and full `798 passed, 1 skipped`
quality gate with zero type errors pass. Update 54000 remains the sole checkpoint
authority, and any remote check or launch requires new exact authorization.

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
9. Separate sessions 1-6 advanced the verified prefix through updates 9000,
   18000, 27000, 36000, 45000, and 54000. The sole continuation source is
   `runs/kaggle/so2_selected_runtime_full_session6_fresh_v1/checkpoints/step_054000.pt`,
   size `16,440,368` bytes, SHA-256
   `2ae4785571e2d1b4e690957e3cf74f749c7e273f1701ee274cc7b2b2e4a8742c`.
   Its partial-target proof and all 13 manifest hashes verify checkpoint and
   metric prefix 54000. Preserve session 4's 67/68 and sessions 5/6's 66/68
   gate caveats, including session 6's one- and five-channel saturation counts.
10. Resume must restore the exact SO2 model, all optimizer groups/state,
   GradScaler, Python/NumPy/Torch CPU/CUDA RNG, named `train_data` and
   `train_corruption` generators, and DDP sampler progress. Require
   `optimizer_step == successful_optimizer_update_count == 54000`, skip prior
   batches by sampler indices without rereading payloads, rebase post-resume
   stochastic streams by rank, and derive LR/beta from the absolute successful
   update count without replaying warmup or advancing on AMP skips.
11. The final continuation uses fresh private kernel
   `maximshtefan/eqvae-so2-selected-runtime-full-session7`, exact datasets
   `maximusshtefan/patches-pre-shuffled-ubc-ocean` and
   `maximshtefan/eqvae-so2-session6-step54000`, T4 GPU, and Internet only for
   the required Torch upgrade. It waits up to 600 seconds and validates the
   exact checkpoint bytes/hash before that upgrade, then runs the unchanged DDP
   trainer through absolute update 60000. A remote check must first establish
   that the session-7 ID has no prior versions; identity checks and launch need
   separate exact authorization.

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
- Session-7 kernel metadata requires the exact ordered `dataset_sources`
  allowlist `['maximusshtefan/patches-pre-shuffled-ubc-ocean',
  'maximshtefan/eqvae-so2-session6-step54000']` and empty competition, kernel,
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

Update 54000 is the sole authority. Its local checkpoint, partial-target proof,
complete 13-entry artifact manifest, model/optimizer/scaler/RNG/generator/sampler
state, two-rank successful metric prefix, validation/fixed-25 coverage, and
accepted 66/68 gate caveat verify. Private dataset version 1 (ID `11716939`)
contains only the exact checkpoint and reports `ready` and `isPrivate=true`.

The existing full resource's version 2 failed before checkpoint load because
that worker did not materialize the correctly declared private input. A fresh
probe configured with both source slugs later verified the private checkpoint's
exact namespaced path, size, and SHA-256; it did not inspect UBC files or train.
Kaggle's internal mechanism remains unknown, so a fresh full resource is a
best-supported workaround rather than a guarantee.

The fresh session-7 transport changes only ID/title, wrapper validation order,
and corresponding builder/guard/test/digest pins. Focused preflight, full
`798 passed, 1 skipped` quality, zero-error type checking, repo/workspace
preflights, and two clean-context adversarial reviews pass. Immediately before
an authorized push, rebuild ignored `run.py` from clean HEAD and require its
embedded manifest to report `git_dirty=false`.

Remote work remains blocked until the user separately authorizes a read-only
identity check proving the session-7 slug has no prior versions. A later exact
launch authorization must name that fresh kernel, the existing private 54k
dataset and checkpoint hash, and continuation through update 60000. GitHub push
is independently unauthorized. Submission alone creates no checkpoint authority.

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
- Full session 4 published complete boundaries 30000, 33000, and 36000. At
  update 36000, clean validation L1/SSIM are `0.06240368277`/`0.7070748183`,
  deterministic-denoising L1 is `0.0655338087`, decoded rotation L2 is about
  `11.66`, and the normalized latent ratio remains about `2.11`. Its checkpoint
  is technically resumable. Gate health is 67/68 due to the documented
  saturated-open F1 row; the user accepted continuing without altering that
  evidence.
- Full session 5 published complete boundaries 39000, 42000, and 45000 before
  cancellation. Its proof/manifest/hash-validated update-45000 checkpoint is
  technically resumable, but its gate health is 66/68 with the two documented
  fully-open F1 rows. The user accepted continuation under this caveat on
  2026-08-18; the next remote upload/launch still requires separate approval.
- Fresh-resource session 6 published complete boundaries 48000, 51000, and
  54000 before cancellation. Its proof/manifest/hash-validated update-54000
  checkpoint is technically resumable. Gate health remains 66/68, but the two
  failed rows now contain one and five saturated-open channels. The user
  explicitly accepted that deterioration on 2026-08-20 after confirming the
  run continued training correctly. The final continuation still requires
  separate exact payload authorization.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/specs/0008-canonical-fixed32-and-remote-debug-tiny-readiness.md`
- `docs/specs/0011-reusable-goal-derived-runtime-and-compiled-fastpath.md`
- `docs/specs/0014-fixed-f01-full-vae.md`
- `docs/specs/0015-fixed-so2-selected-runtime-readiness.md`
- `src/eqvae/training/selected_runtime_runner.py`
- `kaggle/kernels/so2_runtime_readiness/run_template.py`
