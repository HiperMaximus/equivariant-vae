# Current Repository Status

Last updated: 2026-08-20

## Fresh-session start here

Read `AGENTS.md`, `GOAL.md`, this file, `docs/specs/README.md`, active Specs
0011-0016, and Decision 0004 completely. The normal-VAE baseline is complete;
the locked Spec 0012 architecture, accepted Spec 0013 mechanics, Spec 0014
fixed 43-convolution SO(2) VAE, and Spec 0015 local selected-runtime integration
and dual-T4 readiness are complete. Spec 0016's batch-25 prelaunch and full
sessions 1-4 are complete through update 36000, session 5 is verified through
update 45000, and the fresh-resource session 6 is verified through update
54000. Keep every downloaded session separate under ignored
directories; never merge outputs into a resume input. The single authoritative
next-resume checkpoint is:

- local path:
  `runs/kaggle/so2_selected_runtime_full_session6_fresh_v1/checkpoints/step_054000.pt`;
- SHA-256:
  `2ae4785571e2d1b4e690957e3cf74f749c7e273f1701ee274cc7b2b2e4a8742c`;
- size: `16,440,368` bytes;
- proof: fresh-resource session 6's `benchmark/checkpoint_resume_proof.json` and matching
  `benchmark/artifact_manifest.json`.

The checkpoint proof's top-level `status=fail` is expected because the target
update 60000 is incomplete. Its
`failure_kind=partial_interval_checkpoint_not_final_resume_proof`,
`latest_checkpoint_step=54000`, `latest_metric_prefix_step=54000`,
`resume_checkpoint=checkpoints/step_054000.pt`, and
`resume_checkpoint_sha256` above are the positive continuation authority. The
artifact manifest has no missing artifacts; all 13 listed hashes match their
downloaded bytes. Both ranks cover successful updates 45001..54000, with four
synchronized AMP-recovery attempts represented by eight rank rows and no
nonfinite successful update. The checkpoint carries model, optimizer, AMP scaler,
Python/NumPy/Torch CPU/CUDA RNG, named generator, scheduler/beta, and DDP
sampler-progress state.

Do not resume from `best_model.pt`, an earlier boundary, any earlier checkpoint
dataset, or an old launcher. Fresh private kernel
`maximshtefan/eqvae-so2-selected-runtime-full-session6` version 1 loaded the
verified update-45000 state, trained through the complete update-54000 boundary,
and terminally reached `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; the freshly
downloaded archive above, not Kaggle's unreliable file-list endpoint, is
authoritative.

The session-4 67/68 caveat remains retained: three of 48 channels in
`encoder_blocks.6.main_gate:f1_radial` were saturated-open on one probe image,
with finite positive gradient/update/precision evidence and a frozen-25 result
showing sample dependence. The session-5 boundary is separately not a clean
scientific pass: 66/68 gate rows pass. One fully-open channel appears in each
of `decoder_blocks.2.output_gate:f1_radial` and
`encoder_blocks.6.main_gate:f1_radial`; both rows retain finite positive
gradient/update/precision evidence. `gate_health_summary.json` reports the
misleading `failure_kind=no_gate_rows` despite the 68-row CSV because the
summary writer uses that label for every non-all-pass result; the CSV is the
actual evidence. Do not weaken, relabel, or conceal either session's result.
Session 6 remains 66/68 at update 54000. The failed rows are still
`decoder_blocks.2.output_gate:f1_radial` and
`encoder_blocks.6.main_gate:f1_radial`, but their saturated-open channel counts
are now one and five respectively. Both retain finite positive gradient/update
evidence and passing precision evidence. Preserve this deterioration exactly;
do not summarize it only as “two gates.”

On 2026-08-18, the user explicitly accepted continuing the 45k lineage under
the session-5 66/68 caveat. That decision retains the session-4/session-5
caveats. On 2026-08-20, after reviewing the 9,000 successful updates, zero
successful nonfinite updates, and finite positive gate gradient/update evidence,
the user explicitly accepted session 6's five-channel deterioration. The final
continuation still needs separate exact dataset-creation and launch authorization.
With exact
authorization, GitHub `main` was pushed at `a4f7a68`; private dataset
`maximshtefan/eqvae-so2-session5-step45000` version 1 (ID `11701181`) was
created and verified private with only `step_045000.pt` at `16,440,368` bytes;
and Kaggle session-6 kernel version 3 was submitted through update 60000. Its
first read-only status is terminal `KernelWorkerStatus.ERROR`. Its authorized
terminal log shows Torch 2.13/CUDA installation completed, then failed before
loading or training because the exact correct path
`/kaggle/input/datasets/maximshtefan/eqvae-so2-session5-step45000/step_045000.pt`
was missing. A subsequent private CPU-only/no-internet mount probe attached
only that dataset and completed: the exact path exists with `16,440,368` bytes
and SHA-256 `703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93`.
This establishes delayed Kaggle attachment provisioning, not a bad upload,
wrong mount convention, checkpoint, model, schedule, or gate-health result;
version 3 still created no checkpoint authority. Its local transport at
`runs/kaggle/so2_session5_resume_dataset` contains only metadata and the
verified checkpoint, and passes the 16-test continuation preflight plus the
full `797 passed, 1 skipped` quality gate. Retry the unchanged transport only
with fresh exact authorization. That authorization was granted and the
unchanged transport was submitted as Kaggle session-6 kernel version 4 through
update 60000. Version 4 is terminal `KernelWorkerStatus.ERROR`: its log proves
Torch 2.13/CUDA installation completed, then the exact 45k checkpoint path was
again absent before checkpoint load or training. The private one-source probe
did mount and hash that exact file, so simple attachment-propagation delay is
not a sufficient explanation. Checkpoint identity/path and the public UBC
source remain valid; the next diagnostic must inspect Kaggle's accepted
full-kernel metadata and reproduce the exact two-source attachment set before
any third full submission. The accepted v4 metadata lists both sources (Kaggle
normalizes their order to private then public); its pulled server script has
the correct private slug/path. An exact T4/Internet, no-Torch/no-training
two-source probe is submitted as `maximshtefan/eqvae-so2-session6-two-source-probe`
v1 later completed with both mounts and the exact 45k bytes/hash, ruling out
the two-source attachment itself. The remaining pre-validation difference is
the full Torch-upgrade preamble (payload extraction writes only under
`/kaggle/working`); its exact no-training probe is submitted as
`maximshtefan/eqvae-so2-session6-preamble-probe` v1 and completed: the
checkpoint exists at its exact size before the upgrade and at its exact SHA-256
immediately after the identical 127-second upgrade, with no wait. Therefore
versions 3/4 are Kaggle full-worker attachment failures, not local path, source
attachment, payload extraction, or Torch behavior. That preamble probe used the
same literal path/hash but did not invoke the full wrapper's checkpoint helper
functions, so do not call it an exact checkpoint-call control. A replacement is
staged locally under `.agent_tmp/so2_session6_exact_checkpoint_probe`: its
generated `run.py` is byte-identical to the full wrapper through line 8922 and
changes only the final entry point. It runs the unchanged Torch upgrade,
payload extraction, `_resume_checkpoint_path()`, mount wait, and byte/hash
validator, then stops before distributed training. Its metadata differs from
the full kernel only by fresh private ID/title, and its local run passes on the
authoritative 45k checkpoint. After exact authorization it was submitted as
private Kaggle kernel `maximshtefan/eqvae-so2-session6-exact-checkpoint-probe`
version 1. It completed and its downloaded proof validates the exact Kaggle
path, 16,440,368-byte size, and SHA-256
`703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93`;
the log contains no distributed-training launch. This proves the existing full
kernel resource—not the checkpoint call path—is the failed coordinate. The authorized
transport-only remedy added a bounded ten-minute read-only mount wait before
the same exact validation. Its wrapper SHA-256 is
`f9210ea3d8fc3b9739d74e0aef69821c4e5bd0af612edb5a6c62743fe91e262c`;
the strict continuation preflight passes 17 tests and the full quality gate
passes. Kaggle session-6 kernel version 5 waited all 600 seconds and terminated
`ERROR` with the input still absent; no checkpoint load or training occurred.
Kaggle's accepted v5 metadata and the successful preamble probe metadata have
identical T4, Internet, Docker-image, and two-source fields. Only their kernel
resources differ: failed full `id_no=131085532`, successful new probe
`id_no=131241843`. The later exact-call probe on another fresh resource also
passes. The existing full-kernel resource is therefore the isolated
provider-side failure coordinate. Any next retry must use a fresh private full
kernel ID with the same transport. That retry is now staged locally as private
kernel `maximshtefan/eqvae-so2-selected-runtime-full-session6`: only the
ID/title and their builder/guard/test pins changed. The checkpoint path, bytes,
SHA-256, dataset allowlist, wrapper flow, model, trainer, resume semantics,
schedule, and update-60000 target remain unchanged. Its wrapper SHA-256 is
`d66ba76f72d7fbbdc85dc991260c6ad55a4ef9f60fb683df31b8201d97e5af63`;
the focused continuation preflight passes 17 tests and the Python quality gate
passes with `798 passed, 1 skipped` and zero type errors. After exact
authorization, Kaggle accepted it as private kernel version 1 from clean commit
`396d897dc442b5e5f9f94e32f01679d35fa69858`. It ended
`CANCEL_ACKNOWLEDGED`; the log proves exact 45k restore and training, and the
downloaded proof/manifest promote only the complete 54k boundary named above.
The final continuation is locally staged without remote writes. Its ignored
dataset directory contains exactly `dataset-metadata.json` and
`step_054000.pt` for private slug
`maximshtefan/eqvae-so2-session6-step54000`; the checkpoint retains the exact
16,440,368 bytes and SHA-256 named above. The known-good private kernel resource
remains `maximshtefan/eqvae-so2-selected-runtime-full-session6`, now pinned to
that dataset/path. Wrapper SHA-256 is
`a66f8134b63169ec9cfe03d621f52141920d92feca9b80d2cca0271fae5c13fd`.
The 17-test continuation preflight and full `798 passed, 1 skipped` quality gate
with zero type errors pass. After exact authorization, private dataset version
1 (ID `11716939`) was created and remotely verified `ready`, `isPrivate=true`,
with only the 16,440,368-byte checkpoint and the hash-pinned description.
Kaggle accepted version 2 of the known-good private kernel through update 60000,
but it ended `KernelWorkerStatus.ERROR` before checkpoint load or training. The
remote metadata contains both exact dataset sources, the pulled remote script
matches local SHA-256
`95a07cd1bf7b26f982fff5d756edf28bdacf3948d72e6d808624525d4eb805af`, and the
checkpoint dataset still reports `ready` with only the expected file. Despite
that, the worker never materialized
`/kaggle/input/datasets/maximshtefan/eqvae-so2-session6-step54000/step_054000.pt`
during the full 600-second wait. This is the same provider-side private-input
mount failure class seen on the old resource, not a checkpoint, source, slug,
allowlist, GPU, or Internet-setting failure. Update 54000 remains the sole
authority. Any probe, fresh-resource workaround, or retry requires new exact
authorization.
Do not accept, store, or use anyone else's Kaggle API key.
Baseline
full-run session 1 is Kaggle kernel version 2 from source
commit `81b5017`; it ended `KernelWorkerStatus.ERROR` after completing the 15000-update
boundary. Its output and checkpoint are verified locally; the FSQ-aligned AMP runtime
and full-output verification corrections are implemented and verified. Session 2 was
Kaggle kernel version 3 from clean source commit `65112aa`; Kaggle closed it with
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` after the complete update-45000/epoch-7.5
boundary. Its output is downloaded and the checkpoint/proof prefix is verified locally.
Final session 3 is Kaggle kernel version 4 from clean source commit `462c9e1`; it reached
`KernelWorkerStatus.COMPLETE` at update 60000/epoch 10.0 from the exact update-45000
checkpoint. Its output, final checkpoint, complete metric coverage, and fixed-25 evidence
are downloaded and verified locally with the accepted single-update session-1 exception.
Preserve any later unrelated or ambiguous work: do not reset, checkout, blanket-restore,
or recreate the tree. Inspect every diff before surgical removal.

Non-equivariant runtime selection is complete. Do not execute old-v2 `p00310` or the
failed-v3 Bmax/main-effects controller. Session-1 source commit `81b5017` and the first
AMP runtime correction are pushed to GitHub; the current HEAD includes the full-output
policy correction.

Baseline full-run session 1 ran on Kaggle as
`maximusshtefan/eqvae-selected-runtime-full`, kernel version 2. The user explicitly
approved its push on 2026-08-10; the guarded API check and push passed. The terminal
status is `KernelWorkerStatus.ERROR`. Logs show successful boundary evaluation and
checkpoint completion at updates 3000, 6000, 9000, 12000, and 15000, followed by an
intentional failure on both ranks when the AMP guard detected an overflow in the
deferred-metrics window ending at update 18000. Output is downloaded under ignored
`runs/kaggle/selected_runtime_full_v2`. Its resume proof names update 15000 and SHA-256
`8f1b2af601354642036d4d71dca8865ea9c7896a71da4ed69f3871559c448f4f`; the local
checkpoint hash matches exactly. Its committed CSV prefix and fixed-25 boundaries also
end at 15000. The fixed-25 output includes originals, reconstruction progress, rotated
inputs and latents at 90/180/270 degrees, error maps, deterministic posterior-`mu`
arrays, first-three-channel views, and PCA views.

The failure was stricter than FSQ, not checkpoint corruption or missing norm clipping.
Both real runner paths passed `observe_skip=False`, so GradScaler skipped/backed off but
the hot loop could not gate successful-step progress immediately; a deferred scale-drop
assertion then killed the run. The update-15000 checkpoint shows scale `1048576`, growth
factor `2`, backoff factor `0.5`, and growth interval `2000`, consistent with normal
dynamic-scale growth/backoff. Gradient clipping was already global norm `1.0` with
foreach. The downloaded rows show one actual two-rank overflow at event label 14007; the
old path let its data/schedule label advance even though GradScaler discarded that
physical optimizer update. The update-15000 checkpoint is finite and loadable but
therefore represents 14999 physical optimizer updates; the user explicitly accepts losing
one or a few updates. The fix sets `observe_skip=True` on eager and compiled paths,
removes the deferred fatal assertion, consumes the next batch after a skip, and does not
advance LR, beta, validation, checkpoint, or successful-update counters. Full-run
summaries and the downloaded-output verifier allow non-finite telemetry only on skipped
rows and still require every successful row to be finite with exact step/rank coverage.
Runtime selection, LR-range, debug, and tiny-overfit gates remain strict zero-skip.
The user locked beta `0.01` on 2026-08-09; do not run an intermediate beta probe.

## Current objective

Spec 0016 is active. It keeps the normal run's exact selected
runtime, seeds, optimizer, corruption/RNG policy, real UBC data, objective,
beta `0.01`, LR schedules, debug/resume bounds, fixed-32 proof, and fixed-25
protocol while replacing only the model with the fixed Spec 0014 `SO2VAE`.
The shared runner now captures actual activation/dtype/gradient/update evidence
for all 34 F0/F1 gates (68 rows) and records a synchronized final 20-update
two-rank real-loader performance window with data wait, step time, VRAM
headroom, and post-settlement compile counters. Normal-model behavior remains
covered by regression tests.

Private `eqvae-so2-prelaunch` version 1 passed debug 4, resume 4→8, and an
independent 128-update fixed-32 proof. The downloaded
prelaunch verdict is fail-closed and bound to a clean source commit plus hashes
of the complete `src/eqvae`, Spec 0001/0016 configs, data selector input,
launcher templates/metadata, lock, and project metadata. The full push guard
also requires explicit measured-cost acceptance. The user accepted the measured
cost and private `eqvae-so2-selected-runtime-full` version 1 ran from the same
commit. Its archive contains exact committed boundaries 3000/6000/9000; the
Kaggle `kernels files` endpoint incorrectly returned `[]`, while the UI and
`kaggle kernels output` exposed the complete 582 MiB archive.

Update 9000 is the accepted session-1 commit point. Its proof hash matches the
checkpoint bytes; schema v5 contains the exact SO2 model, 421 optimizer-state
entries across three groups, GradScaler, Python/NumPy/Torch CPU/two-rank CUDA RNG,
named `train_data`/`train_corruption` generators, and DDP sampler progress. Config,
effective-config, selected-runtime, row-ID, and policy-ID identities match the
session-1 payload and current execution core. The committed CSV has both ranks for
updates 1..9000, two synchronized isolated AMP skips with immediate recovery,
zero nonfinite successful rows, validation/fixed-25 boundaries at 3000/6000/9000,
and 68 passing gate rows. Partial summary `status=fail` is expected because the
60000-update experiment is unfinished.

Session-2 transport follows the successful normal-VAE pattern exactly. Ignored
`runs/kaggle/so2_session1_resume_dataset` contains only dataset metadata and the
verified checkpoint for private slug
`maximusshtefan/eqvae-so2-session1-step9000`. The continuation wrapper pins the
exact Kaggle path/hash, attaches only UBC plus this SO2 dataset, and passes
`--resume`; the shared trainer/checkpoint/sampler/schedule implementation is
unchanged. The continuation guard validates the original clean prelaunch/session-1
execution core while allowing only the full wrapper/metadata transport change.
The wrapper and archive verifier received independent adversarial review with
zero remaining findings. The final local preflight passes 16 continuation tests;
the focused resume/archive mutation suite passes 18 tests; and
`./scripts/python_quality.sh` passes Ruff, BasedPyright, and 797 tests with one
expected GPU-only skip. Commit `475e215` is pushed to GitHub. After exact user
authorization, private Kaggle dataset
`maximusshtefan/eqvae-so2-session1-step9000` (dataset ID `11656723`) was created
with only `step_009000.pt`; Kaggle reports the exact `16,440,368`-byte size and
`isPrivate=true`. The guarded kernel push passed locally but Kaggle rejected it
before creating a run because the 30-hour weekly GPU quota is exhausted
(`42.00h` used, `0.00h` remaining, `refreshAt=2026-08-15T00:00:00`). The quota
then refreshed to `30.00h` remaining. From clean source commit `325b5db`, the
continuation preflight passed again and private Kaggle kernel version 2 was
successfully pushed at 2026-08-14 19:11 COT. It ended
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; the complete 583 MiB output is downloaded
separately under ignored `runs/kaggle/so2_selected_runtime_full_v2_session2`.
Resume restored update 9000, epoch 1.5, LR `0.000951958`, and scaler `65536`, then
committed boundaries 12000/15000/18000. The worker began boundary 21000 but
cancelled before its validation/checkpoint completed, so 18000 is the only valid
commit point. Its schema-v5 checkpoint hash matches all proof/manifest entries;
model/optimizer/scaler/RNG/generator/DDP sampler state is complete. CSV coverage
is exact for both ranks over successful updates 9001..18000, with three isolated
synchronized AMP skips, zero nonfinite successful rows, 12 validation rows, 54
fixed-25 equivariance rows, and 68 passing gate rows. Fixed-25 originals match
session 1 byte-for-byte. Clean validation L1 improved from about `0.07148` at
update 9000 to `0.06813` at 18000; deterministic-denoising L1 improved from
about `0.07520` to `0.07107`. Kaggle quota is `18.00h` remaining. The next action
requires explicit authorization to upload the exact update-18000 checkpoint to a
new private dataset and push session 3; do not reuse the update-9000 dataset. The
user supplied that exact authorization. Ignored
`runs/kaggle/so2_session2_resume_dataset` contains only dataset metadata and the
verified `step_018000.pt`. The session-3 wrapper/metadata/guard pin the exact
private slug, mount path, hash, step/config/runtime identities, session-2 clean
payload authority, and unchanged execution core. Local preflight passes 16
tests; the focused resume/archive suite passes 18 tests; the full quality gate
passes Ruff, BasedPyright, and 797 tests with one expected GPU-only skip. Commit,
private dataset publication/verification, and guarded launch remain. Commit
`c8ff951` is pushed to GitHub and its clean embedded payload passes preflight.
After the user supplied the exact required authorization, private dataset
`maximusshtefan/eqvae-so2-session2-step18000` (ID `11665702`) was created with
only `step_018000.pt`; Kaggle reports the exact `16,440,368`-byte size,
`isPrivate=true`, and the pinned hash in its description. From clean source
commit `d251175`, the continuation preflight passed again and private Kaggle
kernel version 3 was successfully pushed at 2026-08-15 12:25 COT. It terminated
with `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` after atomically flushing complete
boundaries 21000/24000/27000. The downloaded output has 9000 successful updates
per rank, five synchronized AMP skips per rank, zero nonfinite successful rows,
12 validation rows, 54 fixed-25 rows, and 68 passing gate rows. All 13 hashes in
the partial artifact manifest match; the immutable fixed-25 originals match
sessions 1-2 byte-for-byte. Update 27000 clean validation is L1 `0.06347796463`,
SSIM `0.7006000355`; deterministic-denoising L1 is `0.06682794668`. The exact
checkpoint-only session-4 continuation targets private slug
`maximusshtefan/eqvae-so2-session3-step27000`. The staged directory contains only
metadata plus the exact checkpoint. The updated wrapper/metadata/guard pass the
16 focused prelaunch/full tests and local package preflight. Transport commit
`694017a` and the exact-approval handoff commit `b05623b` are on GitHub. After
payload-specific authorization, private dataset ID `11676466` was created with
only `step_027000.pt`; Kaggle reports the exact `16,440,368`-byte size,
`isPrivate=true`, and the pinned hash in its description. The package was rebuilt
from clean commit `b05623b`; guarded Kaggle kernel version 4 terminated
`KernelWorkerStatus.CANCEL_ACKNOWLEDGED` after complete boundaries
30000/33000/36000. All 13 partial-manifest hashes match; the schema-v5
`step_036000.pt` hash is
`4001c45c023d380f857c8b3e548a314c06a48f270d02529f6dabb875f4b209eb`.
Each rank has exactly 9000 successful updates, four synchronized AMP skips, and
zero nonfinite successful rows; validation/fixed-25 coverage is complete and
originals remain byte-identical. Clean validation L1/SSIM at update 36000 are
`0.06240368277`/`0.7070748183`; deterministic-denoising L1 is `0.0655338087`.
Gate health is 67/68: `encoder_blocks.6.main_gate:f1_radial` has three
saturated-open channels on the one-image capture despite finite positive
gradients/updates and passing precision evidence. Do not silently waive or
rewrite this locked diagnostic. The user accepted continuing with this caveat on
2026-08-17.

### Historical session-5 launch record (do not follow as a handoff)

The following session-5 transport, probes, and launch record is retained only
as evidence. It is superseded by the update-45000 session-6 authority at the
top of this file; no fresh agent may follow its update-36000 procedures.

| Session | Separate raw download | Last committed checkpoint | SHA-256 | Checkpoint dataset used by next session |
| --- | --- | --- | --- | --- |
| 1 | `runs/kaggle/so2_selected_runtime_full_v1_session1` | `step_009000.pt` | `1f53fe16aecf6382bf450cd0ac2be5db9fe2bbe6405dfcaa2c196cb40bca8e7d` | `maximusshtefan/eqvae-so2-session1-step9000` (ID `11656723`) |
| 2 | `runs/kaggle/so2_selected_runtime_full_v2_session2` | `step_018000.pt` | `5911ad37a1ed3f8a92055e45717be496d18545426e56667e1989a3da9a525ec4` | `maximusshtefan/eqvae-so2-session2-step18000` (ID `11665702`) |
| 3 | `runs/kaggle/so2_selected_runtime_full_v3_session3` | `step_027000.pt` | `7adfea7850ee7ab620f0363ca4a8fe9e41fd67160feeaeae1f07ff291a0bf6ba` | `maximusshtefan/eqvae-so2-session3-step27000` (ID `11676466`) |
| 4 | `runs/kaggle/so2_selected_runtime_full_v4_session4` | `step_036000.pt` | `4001c45c023d380f857c8b3e548a314c06a48f270d02529f6dabb875f4b209eb` | Verified private dataset `maximshtefan/eqvae-so2-session4-step36000` version 1; session-5 kernel v1 failed before training because its input was not mounted |
| 5 | `runs/kaggle/so2_selected_runtime_full_v5_session5_remote` | `step_045000.pt` | `703dc15aeca96235227780cbea0a35b918faa404ec42fda701324a1ae17abd93` | Private `maximshtefan/eqvae-so2-session5-step45000` version 1 (ID `11701181`); session-6 kernel v3 submitted |

Every raw output and resume-staging directory is gitignored. The table records
local state; verify the files and proofs themselves before transport. On
2026-08-17, the user identified `maximshtefan` as the independently authenticated
operator. The guarded fresh-OAuth read path listed all four expected public UBC
source files. The local session-5 transport now pins that kernel/checkpoint
owner, `step_036000.pt`, its exact hash and `16,440,368` bytes; it passes the
focused suite and full-kernel preflight.

At 2026-08-17 17:46 COT, private dataset
`maximshtefan/eqvae-so2-session4-step36000` became ready as version 1 and its
only remote file was `step_036000.pt` at `16,440,368` bytes. Clean commit
`2b41524ee26561c412fb6078e9138faa3fbb4fea` is pushed to GitHub; its clean
embedded package passed preflight and full quality (`797 passed, 1 skipped`).
Kaggle session-5 kernel version 1 reached terminal
`KernelWorkerStatus.ERROR` at 2026-08-17 18:40 COT before training started.
Its log reports that the ready checkpoint dataset was not mounted at
`/kaggle/input/eqvae-so2-session4-step36000/step_036000.pt`; the same worker
also could not resolve PyPI while attempting the required Torch upgrade. The
remote dataset still lists the one verified checkpoint, so this is a Kaggle
attachment/worker-network failure, not checkpoint corruption or new gate-health
evidence. Preserve the session-4 lineage and 67/68 caveat. Do not retry, alter
the trainer, or select another checkpoint without fresh exact authorization.

The authorized no-GPU/no-internet input probe
`maximshtefan/eqvae-so2-session5-input-probe` version 1 completed and its raw
ignored artifact at `runs/kaggle/so2_session5_input_probe_v1` records
`exists=false`, `bytes=null`, and `matches_expected=false` for that exact
expected checkpoint path. It imported neither Torch nor training code. This
independently confirms absence at the declared mount path. Its authorized
version-2 listing records `/kaggle/input` as exactly `['datasets']`, identifying
Kaggle's alternative input-root convention. Its version-3 listing records
`/kaggle/input/datasets` as exactly `['maximshtefan']`; the dataset-slug leaf
was identified by version 4 as `eqvae-so2-session4-step36000`. The exact
checkpoint mount is therefore
`/kaggle/input/datasets/maximshtefan/eqvae-so2-session4-step36000/step_036000.pt`.
The training launcher still uses the legacy non-namespaced path. A future
transport-only correction must change only that path and its exact guards; it
requires fresh authorization before a new kernel push.

The separate authorized fresh-worker probe
`maximshtefan/eqvae-so2-torch-cuda-probe` version 1 requested T4 plus Internet
without data, Torch import, or installation. Its ignored raw report at
`runs/kaggle/so2_torch_cuda_probe_v1` records `nvidia-smi` absent and the PyPI
`pip install --dry-run --upgrade torch torchvision torchaudio` timing out after
180 seconds. Thus the probe worker was not CUDA-provisioned and did not have
usable PyPI access; this is independent of, and blocks a retry after, the mount
path correction.

On 2026-08-17, the authorized no-data probe version 2 completed after an
explicit Kaggle CLI `--accelerator NvidiaTeslaT4` request. Its ignored report at
`runs/kaggle/so2_torch_cuda_probe_v2` again records `nvidia-smi` absent and the
same 180-second PyPI dry-run timeout; its returned log has no allocation or
quota explanation. Thus neither metadata nor the explicit CLI accelerator flag
obtained a CUDA worker for this account/kernel, but the precise Kaggle-side
reason is unproven. Do not correct or retry the full transport until the Kaggle
UI exposes an enabled GPU choice and a new no-data probe returns CUDA and PyPI
success.

On 2026-08-17, version 3 of the same private, input-free probe explicitly
requested `NvidiaTeslaT4`, disabled Internet, and imported the preinstalled
Torch only. Its ignored report at `runs/kaggle/so2_torch_device_probe_v3` is
unambiguous: `torch_version=2.10.0+cpu`, `torch_cuda_version=null`,
`cuda_available=false`, and zero devices. The user added account verification
after this worker started, so it cannot establish the post-verification state.
After the Kaggle UI shows an enabled GPU choice, obtain fresh exact authorization
for one new no-data device probe before any mount correction or full retry.

The user granted that authorization after completing account verification.
Version 4 passed: its ignored report at
`runs/kaggle/so2_torch_device_probe_v4` records `torch 2.10.0+cu128`, CUDA
`12.8`, `cuda_available=true`, and exactly two `Tesla T4` devices. CUDA
provisioning is therefore healthy on the post-verification account; the only
remaining worker-readiness evidence is a freshly authorized no-data PyPI
Torch-upgrade dry-run. Do not yet correct/retry the full transport.

The user authorized that exact post-verification PyPI check. Version 5 passed:
its ignored report at `runs/kaggle/so2_torch_pypi_probe_v5` has return code zero
for the bounded `pip install --dry-run --upgrade torch torchvision torchaudio`,
resolving CUDA-enabled Torch `2.13.0` and its CUDA-13 dependencies. CUDA and
Internet readiness are now proven. The transport-only mount correction is local
commit `3d5bf766f323645d725f92a9dd5e27deaf438b7b`: it changes only the checkpoint
path to `/kaggle/input/datasets/maximshtefan/eqvae-so2-session4-step36000/step_036000.pt`
and repins the wrapper SHA-256 to
`a01eb989866b1bb2e8017ef04a2fdb0de9fe6c4f6c4a934c80a567813fc0d4e4`.
The full continuation preflight passes with all 16 focused tests and a clean
embedded manifest; the complete quality gate is `797 passed, 1 skipped`.
The user then authorized the exact retry. GitHub `main` contains clean commit
`e1b9e9f9a28299f4604a768720345ae9cd7c2fb3`; Kaggle session-5 kernel version 2
was submitted with only the public UBC and exact private step-36000 datasets.
Its first read-only status at 2026-08-17 21:28 COT is
`KernelWorkerStatus.RUNNING`. Do not poll in turn or infer a new checkpoint:
download and validate only a later terminal session's own proof and manifest.

1. Completed: recomputed the checkpoint SHA-256 and size and cross-checked
   session 4's checkpoint proof and artifact manifest.
2. Completed to the safe observable extent: the user identified
   `maximshtefan`; its global fresh-OAuth path listed the public UBC source.
   Never receive, print, install, or inspect credentials.
3. Completed locally: chose the new private checkpoint destination
   `maximshtefan/eqvae-so2-session4-step36000`. Obtain new payload-specific user
   authorization that names that exact slug, `step_036000.pt`, the SHA-256 above,
   and the session-5 launch. Prior session authorizations do not carry forward.
4. Completed locally: staged ignored
   `runs/kaggle/so2_session4_resume_dataset` containing exactly
   `dataset-metadata.json` and the verified `step_036000.pt`; no metrics,
   earlier checkpoints, normal-VAE files, or generated artifacts belong there.
5. Completed locally: update every transport pin together:
   `kaggle/kernels/so2_selected_runtime_full/run_template.py`, its
   `kernel-metadata.json`, `scripts/kaggle_kernel.sh`, and
   `tests/test_so2_full_run.py` (plus the builder's owner discriminator). Pin the new owner/kernel ID,
   checkpoint-dataset slug, Kaggle mount path, filename, step `36000`, exact
   hash/size, wrapper digest, two-dataset allowlist, and unchanged session-1
   execution-core authority. Do not change the trainer, model, runtime plan,
   schedule, or checkpoint schema.
6. Completed locally: regenerated the ignored `run.py`, ran the focused continuation
   tests and `./scripts/kaggle_kernel.sh preflight-so2-selected-runtime-full`, inspected
   the ZIP/manifest, ran both repo/workspace preflights, passed `git diff --check`, and
   passed `./scripts/python_quality.sh` in a persistent terminal (797 passed, 1 skipped).
   Clean commit `2b41524` is pushed and its embedded manifest is clean.
7. Completed remotely under the exact authorization: the authenticated operator created
   and verified the private checkpoint dataset, then pushed session-5 kernel version
   1. It targets absolute update `60000`; never cap it at `45000`. Version 1 stopped
   before trainer/DDP execution because Kaggle did not mount that ready checkpoint
   dataset, and its worker also lacked PyPI DNS. No training output or metric prefix
   was produced.
8. With fresh exact authorization, first establish why Kaggle did not attach the
   verified private dataset at its expected path. The minimal probes show that
   the legacy root is absent and the worker instead exposes
   `/kaggle/input/datasets/maximshtefan/eqvae-so2-session4-step36000`; the exact
   checkpoint path is the same leaf plus `step_036000.pt`. After the
   transport-only mount-path correction, first obtain fresh positive T4/CUDA and
   PyPI-resolution evidence; do not retry while the fresh-worker probe reports
   neither. Then preflight and obtain fresh exact authorization before a new
   kernel push. Only a later completed session's separately downloaded
   `checkpoint_resume_proof.json` may name a new commit point; 39000/42000/45000
   are merely possible boundaries, never assumptions.

Spec 0015 is complete. Its single guarded remote coordinate passed. Registry
kind `so2_vae_fixed` accepts no architecture
options and fails closed on exact `SO2VAE` identity, latent width 16, 43 learned
convolutions, 34 radial gates, and `1,180,035` parameters. The shared runner now
selects that kind without changing normal-model behavior. The one-use readiness
executor pins the exact selected-runtime artifact hash and complete FP16 /
Inductor / compiled-autograd / Python-reducer DDP / channels-last / fused-AdamW
bundle. It uses two rank-local generated `1x3x256x256` inputs and no dataset,
proves precompile buffer identity, DDP gradient averaging and parameter sync,
zero-head then named upstream updates, finite settled execution, and captures
actual FP16 F0/F1 activation evidence with FP32 gate math for exactly 68 rows.
The strict downloaded-artifact validator cross-checks both rank records,
aggregates, exact proof bodies, and exact 34-module-by-two-family identities.

Post-review `./scripts/python_quality.sh` passes Ruff format/check, 780 tests
with one expected GPU-only skip, and BasedPyright with zero errors. Two fresh
read-only reviews covered runtime/DDP/compile/AMP/optimizer correctness and
gate/evidence/performance/scope correctness. Their plan pinning, executed gate
semantics, gradient/update, cross-rank counter, validator, identity, generated
launcher, and mutation-test findings were fixed; both reviewers rechecked with
zero unresolved findings. Local kernel preflight and `git diff --check` pass.
Private Kaggle kernel `maximusshtefan/eqvae-so2-runtime-readiness` version 1
completed from clean source commit `6cdccb0`. The downloaded strict validator
passes. Both Tesla T4 ranks used Torch `2.13.0+cu130` / CUDA `13.0`, batch 1,
the exact selected runtime, and generated device-resident inputs with no data
source. The result has zero AMP skips, nonfinite losses/parameters, settled
graph breaks, recompiles, buffer divergence, DDP mean error, or parameter
divergence. The zero head and all named decoder/posterior/encoder/stem/F0/F1
updates pass. All 68 actual F0/F1 rows pass with positive finite gradient and
update evidence and no dead channels. Diagnostic median settled step time is
`132.285 ms`; peak allocated/reserved memory is `410.016/538 MiB`, leaving
`96.392%` reserved-memory headroom. The compact hash-bound summary is
`docs/data/spec0015_so2_runtime_readiness_v1.json`; raw output remains ignored
under `runs/kaggle/so2_runtime_readiness_v1`. The pip CUDA-13 upgrade reported
conflicts with unused preinstalled CUDA-12 RAPIDS packages, but the installed
Torch/CUDA identity and complete probe both passed. No rerun or training was
launched.

Spec 0014 is complete locally. `src/eqvae/models/so2_vae.py` mirrors all 43
normal-VAE convolution positions with the locked `9-low` stem, `7-low`
remainder, A/B/C/D equal-copy layouts, scalar RGB/latent boundaries, six
branch-local downsamplers, and six branch-local upsamplers. The instantiated
model is exactly `1,180,035` learned parameters with the locked
`1,172,304/3,600/4,096/35` coefficient/norm/gate/bias partition. Eager
two-step gradient-driven updates, optimizer grouping, CPU autocast, base and
autocast fullgraph reuse, deployment shapes, contraction call counts, and
cardinal/non-cardinal endpoint evidence pass. The readiness workstream stops
here. Any real-data debug or full training requires separate explicit
authorization.

Spec 0012's small non-training basis oracle is implemented and measured. The
tracked manifest selects the fixed F0/F1 contingency (`F01`), not an F2 model.
Selected global profiles are `7-low=(r=[1,1.90395977,2.75], sigma=[.3,.3,.3],
qmax=[2,2,2])` and `9-low=(r=[1,1.99907757,2.87711643,3.75],
sigma=[.3,.3,.3,.3], qmax=[2,2,2,4])`. `9-full` also passes raw rank,
conditioning, perturbation, and `escnn` span checks at
`r=[2,2.64125343,3.25768854,3.75]`, all widths `.3`, all qmax `4`, but its
incremental high-order sampled-grid error is `E_high=2.0000585868`, above the
locked `E_limit=0.1702658838` (`D_high=24`, `E_floor=0.1135105892`), so Spec
0012 requires F2 rejection. The audit records every pair/angle error: exact
90-degree transforms are at most `2.74e-14`, while the identified worst case
is `F2->F2` at 45 degrees. The incremental q3/q4 subspaces match `escnn` within
`8.01e-8` projector distance before that decision.

The locked `7-full` search premise is internally incompatible: q4 must appear
on two shells with `r>=2`, but below the 7x7 bound `2.75` the coarse grid offers
only `2` and `sqrt(5)`, separated by less than `.25`; hence it supplies zero
legal seeds even though the continuous point `[1,2,2.75]` with qmax `[2,4,4]`
is feasible. The oracle records this premise failure and does not invent a
seed. It does not affect the chosen result because 7x7 adequacy requires an
adequate 9x9 high-order reference, and `9-full` fails that prerequisite.

The user locked equal **representation-copy** capacity, not equal packed tensor
width. At baseline logical widths `[32,48,64,96]`, the fixed F0/F1 copy pairs
are `[(16,16),(24,24),(32,32),(48,48)]`; packed widths are
`[48,72,96,144]`. The fixed `16F0` latent and scalar RGB interfaces are
unchanged. The targeted layout refresh now records `1,172,304` basis
coefficients, `3,600` normalization parameters, `4,096` gate parameters, and
`35` scalar biases: `1,180,035` total learned parameters under the `3,958,435`
cap. Dense-convolution MACs are `159,837,585,408` per sample and expansion MACs
are `159,453,168` per forward. Dense learned-convolution compute is therefore
4.383x the baseline even though learned parameters are only 29.81% of its cap;
the comparison is parameter-bounded, not compute-matched.

`scripts/select_so2_basis.py --refresh-layout` loads the locked profiles and
recomputes only layout-dependent counts and initialization evidence. The
128-trial comparison covers 13 distinct layer signatures and 25 frequency
outputs; ratios span `0.9977233322..1.0079137704`, all inside `[0.9,1.1]`.
The compact manifest retains only selected `7-low`/`9-low` plus the fixed field
handoff; rejected profiles remain only in the audit. Hashes of the audit's
radial search, profile, escnn-reference, high-order/F2, and locked-premise
sections are unchanged.

The fixed Spec 0013 local probe is now implemented. It hard-codes only the
selected F0/F1 layouts and pair banks, expands FP32 master coefficients with
the selected padded `torch.bmm` for hidden maps (fixed `torch.mm` at scalar
boundaries), performs direct static assembly and exactly one dense `conv2d`,
keeps basis/layout construction outside forward, and adds only the
locked norm/gates, identity block, encoder/decoder transitions, resampling, RGB
interfaces, and scalar latent heads. It does not assemble or expose the
43-convolution VAE. Exact eventual counts remain `1,180,035`.

Focused local verification passes: 69 tests with 329 pinned-escnn/SciPy
deprecation warnings, Ruff format/lint, BasedPyright, exact check-only
128-trial layout refresh, generated-kernel local preflight, and agent preflight.
All pair banks and multi-copy signatures match the pinned escnn reference;
every one of 40 profile/pair/angle sampled-equivariance rows passes
`ours <= max(5e-4, 1.10*escnn)`. Reduced FP64 gradients,
generalized-He/bias/count checks, norm/gates, residuals, resampling, RGB/latent,
eager optimization, and CPU fullgraph pass. Exact evidence and source hashes
live in `docs/data/spec0013_so2_cpu_probe.json`. The earlier full-suite result
remains 701 tests with one expected GPU-only skip; the suite was not rerun
because this slice touched no shared infrastructure.

Fresh mathematical review found no premise error and required complete named
coefficient-gradient, pair/angle, initialization/bias/count, and source-binding
coverage; all were added. Fresh compile/performance/scope review required the
exact selected-runtime JSON hash and live compiler/DDP readbacks, batch-4
fullgraph and initial-break evidence, load-bearing AMP/finiteness, recorded raw
timing and CV diagnostics, DDP-wrapped matched timing, and compiled-FP16
assembly timing; all were added. The generated runner was rebuilt after the
final source changes.

The reviewed local slice was committed and pushed as `e57f086`. With explicit
permission, private Kaggle kernel version 1 ran the exact selected Spec 0011
bundle on two T4s and produced a valid fail artifact, now summarized at
`docs/data/spec0013_so2_dual_t4_probe_v1.json`. The runtime transfer was healthy:
32 settled updates, zero AMP skips/nonfinite values/graph breaks/recompiles,
matching cross-rank buffers and DDP mean/update reference, `797/950 MiB`
allocated/reserved, `0.43..0.92x` compiled/eager, `1.02..2.04x` EQ/normal, and
`1.828x` topology-weighted EQ/normal.

Private Kaggle version 2 ran from clean commit `afec7af`; its reviewed compact
summary is `docs/data/spec0013_so2_dual_t4_probe_v2.json`. The corrected
GradScaler diagnostic resolves v1's false decoder alarm: the worst output and
coefficient-gradient relative RMS values are `0.00061934` and `0.00066145`,
comfortably below `0.005/0.02`. DDP, AMP, compile, buffer, VRAM, compiled/eager,
EQ/normal, and corrected-control CV gates pass. Bitwise equality remains
diagnostic only; no numerical correctness gate failed.

V2 validly rejected all three predeclared mechanics on the old isolated `0.10`
assembly-fraction gate. On 2026-08-13 the user accepted replacing that
microbenchmark gate with experiment-relevant per-block runtime checks and
selected the fastest measured path: padded `bmm` plus direct assembly. The old
`1.828x` "topology-weighted" v1 aggregate is also not a final gate because four
multi-convolution probe blocks do not map exactly onto all 43 positions.

Spec 0013 is accepted and complete. Kaggle kernel v3 from `c823a7e` passed every
correctness, compiled/eager, EQ/normal, AMP, DDP, graph, and VRAM gate. It
originally reported fail only because the per-window 10% timing-CV diagnostic
was still load-bearing. Maximum output/gradient errors were
`0.000619/0.000662`; compiled/eager ratios were `0.385..0.721`, EQ/normal ratios
were `1.118..2.014`, parameters matched exactly across ranks, and reserved
memory was `954 MiB`. CV failures mirrored across ranks: encoder window 0,
D-to-D window 1, and the decoder normal-control pool. The tracked summary is
`docs/data/spec0013_so2_dual_t4_probe_v3.json`.

On 2026-08-13 the user selected compiled execution, made raw CV diagnostic, and
accepted the existing evidence without a rerun. Padded `bmm` plus direct
assembly is the fixed mechanics. Eager timings remain a reference; compiled
medians and the matched compiled normal controls are the performance evidence.
Do not add another mechanics arm or runtime option. Spec 0014's authorized
full-VAE assembly is complete; selected-runtime readiness is the next separate
authorization boundary.

This remains one-off experiment code. Do not ship runtime architecture,
support, radial, field-layout, or group options; do not retain rejected
candidates as selectable model branches. Optimize the simplest singular path
for the locked layouts, fixed shapes, current Torch runtime, and dual-T4 target;
do not add abstractions, fallbacks, portability, or generalized shape handling
without a measured need. Prefer a small direct Kaggle check when the alternative
is speculative local overengineering. Bitwise parity is not required: the explicit Spec
0013 numerical and sampled-equivariance tolerances are the contract, including
ordinary AMP/compiler rounding and the documented finite-grid resampling floor.
For the eventual EQ convolution, all radial/trigonometric sampling, masks,
field offsets, legal-pair selection, QR coordinates, and basis buffers are
resolved offline or in `__init__`. Training `forward` contains only fixed-shape
coefficient-to-pair-block contractions, static block assembly, and one dense
`conv2d`; it performs no manifest parsing, basis generation, pair discovery,
SVD/QR, or adaptive branching. Kernel expansion itself must remain in training
forward because learned coefficients change after each optimizer step.
The ignored `reference/escnn` checkout is test-only. The existing venv can run
its SO(2) oracle without dependency sync by using the Spec 0012 no-cache
`joblib.Memory` shim and fail-loud `lie_learn` SO(3) sentinel; never expose that
bootstrap to training or the general project import path.
The completed control's fixed blur+stride-2 downsample has an even-grid
90-degree phase error. The first EQ comparison retains that exact fieldwise
operator, accepting and reporting the sampled-grid limitation rather than
retraining the completed normal control. The phase-centred fixed 6x6 repair is
reserved for a later matched rerun. Lock primitive and non-cardinal tests before
recording error magnitudes.

Spec 0011 is now a lean, two-architecture Kaggle tuning campaign. For correct dual-T4
`drop_last=True` training, minimize

```text
floor(real_train_patch_count / global_batch)
* synchronized_mean_steady_state_step_wall_time
```

Recipe and integer batch are selected jointly. Largest feasible batch, step latency,
throughput alone, compile time, and the fewest enabled options are not objectives. A fast
correct recipe may retain neutral, redundant, or inert toggles.

## Direction correction

The user rejected the uncommitted Spec 0011 v4 implementation because it had become an
audit-grade general platform for a configuration search that will be used for only the
non-equivariant and continuous-`SO(2)` models. The active spec was replaced by v5 on
2026-08-08.

V5 keeps only what can change or establish the selected training configuration:

- reviewed complete recipe bundles from repository/Torch fast paths;
- joint nonmonotone recipe×batch measurement;
- correct dual-rank update/timing checks;
- coordinate-local OOM and focused failure repair;
- one CSV and summary JSON per direct Kaggle probe;
- fresh finalist confirmation and a real-loader check;
- a concrete selected-runtime config fragment.

It drops exhaustive internal/source inventory, formal all-pairs coverage, independent
duplicate verification, generalized transformation DAGs, capsule/cache equivalence
certificates, and broad audit-mutation machinery. Neutral options do not need ablation.

## Immutable v2 evidence

`docs/data/spec0011_runtime_recipe_v2/` retains 309 immutable rows and its producer. Use
the rows as explicit priors for option ordering, failure modes, VRAM, timing regions, and
batch wells. The incomplete `p00310` is permanently unschedulable. Old rows do not prove
performance under a newly upgraded runtime.

## Rollback boundary

Proven useful and preserved:

- existing training/fastpath and selected-runtime work;
- latest-PyPI Torch upgrade support;
- immutable v2 evidence;
- guarded Kaggle packaging and atomic publication foundations;
- unrelated dirty repository changes.

Removed as v4-only overbuild:

- the activation/controller/identity/inventory/measurement/independent-verifier stack;
- the v4 maximal-cover/statistical policy and audit-focused tests;
- v4 artifact-parent, certificate, and executor-readiness packaging assumptions.

The large failed-v3 controller and the later one-use direct probe kernel are removed. Do
not restore their exact-Bmax, main-effects, beam/frontier, exhaustive-audit, certificate,
or bespoke packaging behavior. The measured winner is retained as one compact config.

## Selected non-equivariant runtime

Use per-rank batch 25 (global 50), conservative FP16 AMP, channels-last, compiled whole
step, Python DDP reducer, compiled autograd, compute/communication reorder, fused AdamW,
TF32, high matmul precision, gradient-as-bucket-view, bucket cap 50 MB, no buffer
broadcast, and foreach clipping. Exact recipe and compact measurement pointers live in
`configs/spec0001/non_eq_vae_runtime_winner.json`.

Fresh batch-25 measurements were 749.8 and 778.7 ms/step, projecting 4499 and 4672 s per
300,000-patch epoch; their mean is about 4585 s (1.27 h). Neighbor batches 18 and 35
projected 4719 and 4954 s; batch 56 projected 5166 s. Baseline AMP projected 10155 s.
VRAM reserved was 6078-6104 MB at the selected coordinate. All selected rows had finite
updates, synchronized ranks, zero AMP skips, zero graph breaks/recompiles after settle,
and zero measured data-wait fraction. Kaggle kernel version 14 is the final confirmation;
do not run more batch probes.

Raw downloaded rows remain ignored under `runs/kaggle/runtime_recipe_probe_v9` and
`runtime_recipe_probe_v14`. The one-use `runtime_recipe_bakeoff` module, Kaggle kernel,
CLI actions, guards, generated launcher, and focused tests were removed. Only the compact
winner JSON and immutable v2 evidence remain.

Exact removed probe surfaces: `src/eqvae/benchmarking/runtime_recipe_bakeoff.py`,
`tests/test_runtime_recipe_bakeoff.py`, `kaggle/kernels/runtime_recipe_bakeoff/`, and all
of their dedicated wiring in `scripts/kaggle_kernel.sh`,
`scripts/build_kaggle_embedded_kernel.py`, `scripts/agent_preflight.sh`, `.gitignore`, and
`tests/test_kaggle_embedded_kernel.py`. The generic latest-Torch policy test remains.

## Multi-session handoff

- Session-2 transport is complete. Private dataset
  `maximusshtefan/eqvae-baseline-session1-step15000` contains only the verified
  `step_015000.pt`; the kernel attaches it beside the UBC patch dataset and rejects a
  missing or hash-mismatched checkpoint before GPU work. Keep this one-off transport;
  do not build a generic layer.
- Use lean checkpoint-only sessions because the projected ~12.7-hour training time
  exceeds Kaggle's 8-hour limit. Every session still targets update 60000 and runs until
  it completes or Kaggle closes it; there is no artificial session cap. Every 3000-update
  boundary flushes metrics, fixed-25 evaluation artifacts, and a resumable checkpoint.
  After cancellation, download the whole session into its own local directory and give
  the next Kaggle worker only the latest fully completed boundary checkpoint. Resume in
  a fresh output directory with index-only loader offsetting and rank-local stochastic
  streams. The checkpoint named and hashed in
  `benchmark/checkpoint_resume_proof.json` is the session commit point. Use
  `latest_checkpoint_step` as the inclusive cutoff; exclude any preflushed CSV rows or
  fixed-25 artifacts above it, and reject a missing/hash-mismatched/non-3000 boundary.
  After update 60000, concatenate the committed CSV prefixes locally by absolute
  optimizer step and choose the global best from downloaded validation results. Keep
  session copies until the merged result is verified; delete redundant copies only
  afterward.
- Session 1 version 2 was pushed from clean commit `81b5017` and ended in error after the
  completed 15000 boundary. Its output is downloaded and verified; do not replace or
  rerun it without new explicit direction.
- Session 2 version 3 was pushed from clean commit `65112aa` at 2026-08-10 11:24 COT and
  ended `CANCEL_ACKNOWLEDGED` after the completed update-45000 boundary. Its output is
  downloaded under ignored `runs/kaggle/selected_runtime_full_v3_session2`. The resume
  proof names `checkpoints/step_045000.pt` with SHA-256
  `e7a0f05e013bff4f7a5bfbfd4442f3c9a6d19cf261c42f54a6d04391be76e88b`; the local
  file hash and loadable schema-v5 metadata match. Fixed-25/validation boundaries are
  complete from 18000 through 45000. The session logged 26 skip rows: 13 synchronized
  two-rank AMP skips, all isolated (maximum consecutive streak one), with zero non-finite
  successful updates. Clean validation L1 improved `0.07733 -> 0.05965` and denoising
  L1 `0.08172 -> 0.06279` from update 3000 through 45000; corresponding SSIM rose
  `0.5690 -> 0.7272` and `0.5647 -> 0.7192`. Visual comparison of the same fixed-25
  originals at updates 3000, 15000, and 45000 shows continued edge/nuclear-contrast
  improvement without collapse: tissue layout, stain, glands, nuclei, stroma, and empty
  regions remain recognizable, with expected VAE smoothing of fine chromatin detail.
  Session-1/session-2 `originals.png` and `originals.pt` hashes match exactly, proving the
  comparison did not replace the fixed examples. All three raw session directories are
  separate and must remain unchanged.
- Session 3 version 4 completed update 60000/epoch 10.0 and is downloaded under ignored
  `runs/kaggle/selected_runtime_full_v4_session3`; the first two session directories are
  unchanged. `step_060000.pt` hashes to
  `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075` and loads as
  schema v5 with update/scaler/model/optimizer/RNG/generator/sampler state intact. Its six
  AMP skips were isolated and recovered immediately. A one-off ignored verification view
  at `runs/kaggle/selected_runtime_full_combined_verified` filters the three committed
  ranges and gives exact 1..60000 two-rank coverage: 120000 successful rows, 38 skip
  rows/19 synchronized attempts, 80 validation rows, 360 equivariance rows, no duplicate
  or missing step/rank. The strict gate's sole blocker is the accepted old session-1
  update-14007 telemetry: both ranks have finite loss, `grad_norm=inf`, zero parameter
  update, and old `amp_step_skipped=0`; every other successful row is finite.
- Primary image evidence is local and ignored, not available on GitHub: fixed originals
  are at `runs/kaggle/selected_runtime_full_v4_session3/artifacts/fixed25/originals.png`;
  final reconstructions, the rotated-input/rotated-embedding grid, and latent PCA/first3
  views are under `artifacts/fixed25/boundary_060000` in that session directory.
- The current local source adds FSQ-aligned rank-0 resume breadcrumbs for data preparation,
  checkpoint-load start, load duration/restored epoch-step/LR/GradScaler scale, and the
  first successful resumed optimizer update with LR/attempt count. It also logs every
  rare AMP non-finite/overflow skip with its consecutive streak and new scaler scale,
  then logs recovery. NCCL initialization also receives the resolved rank-local CUDA
  device explicitly, removing PyTorch 2.13's ambiguous-barrier-device warning. Session 3
  exercised these changes successfully; version 3 predates them.

The user prefers direct, bounded Kaggle experiments over defensive local machinery and is
comfortable with liberal probe pushes. Still use the repository's `KAGGLE_*_CONFIRMED`
guards; never infer permission for the full training launch from probe permission.

`configs/spec0001/non_eq_vae_selected_runtime.json` is a hash-linked runtime snapshot.
Its `full_training_launch_ready=false` and probe-era blocker strings record creation-time
state; do not casually edit them and invalidate downloaded runtime/debug evidence. Live
readiness and the remaining multi-session blocker are recorded here.

## Verification state

Spec 0016 local verification passes both generated-kernel preflights, 244
focused/shared-runner regression tests, `git diff --check`, Ruff format/check,
794 full-suite tests with one expected GPU-only skip, and BasedPyright with zero
errors. The prelaunch/full packages import from their embedded payloads, attach
only the real UBC dataset, reject normal checkpoint lineage, and keep remote
prelaunch/full writes permission-gated. Remote batch-25 feasibility, learning,
and measured epoch/session cost remain intentionally unproven until the one
authorized prelaunch runs.

Spec 0015 acceptance is complete: the final post-review quality gate is
780 passed, 1 expected GPU-only skip, with Ruff formatting/check and
BasedPyright at zero errors. The dedicated embedded-kernel preflight verifies a
private dual-T4 kernel with empty dataset, competition, kernel, and model source
lists. Two clean-context adversarial reviews have zero unresolved findings.
Private Kaggle v1 completed and its exact JSON/68-row CSV pass the strict local
validator. The fixed SO2 model is operationally ready at the single batch-1
dual-T4 coordinate. This is not a training-performance claim. No real data or
training is authorized.

Spec 0014 focused verification passes 84 analytic-basis/primitive/kernel/model/
optimizer tests; its 11 full-model tests cover exact topology/counts, complete
branch order and shapes, one-conv/bmm/mm contracts, two-step finite gradients
and updates, AMP-facing state, base/autocast fullgraph reuse, independent decoder
equivariance, raw transform floor, and accepted fixed-downsampler phase error.
Fresh mathematical/topology and compile/performance/scope reviews pass after
their acceptance-test findings were corrected. No Kaggle probe was run because
the remaining hardware question belongs to selected-runtime integration, not
model assembly. Final `./scripts/python_quality.sh` passes Ruff formatting/check,
767 tests with one expected GPU-only skip, and BasedPyright with zero errors.
Both repo/workspace preflights and `git diff --check` pass.

Kaggle LR-range kernel v1 passed: 192/192 two-rank updates from `2e-5` to `3e-3`, zero
AMP skips/non-finites, and smoothed loss `0.645 -> 0.251`. Manual curve inspection chose
effective `7.216878e-4` for tiny overfit and peak `1e-3` for scheduled full training; the
automatic `2e-5` recommendation was rejected as startup-noise-biased.

Kaggle debug/tiny kernel v6 passed. Resume loaded update 4 and continued to update 8.
The fixed-32 check completed exactly 128 updates on both ranks with zero skips/non-finites;
smoothed L1 improved `0.2388 -> 0.1717` (28.1%) and reconstruction loss improved
`0.3048 -> 0.2361` (22.5%). The strict downloaded-output verifier passed.

The stricter clean-memorization probe removes corruption, latent sampling, and KL
(`beta=0`, deterministic `z=mu`) while retaining the selected runtime/LR. Kaggle v7
stopped before training because the zero corruption probability did not match the named
locked profile; the direct fix added the explicit `no_corruption_probe` profile. V8 then
completed 512 two-rank updates with zero skips/non-finites. Smoothed L1 improved
`0.2416 -> 0.0930` (61.5%) and reconstruction loss `0.3076 -> 0.1451` (52.8%); the final
64-step block still descended, so the network clearly learns but has not met the stricter
memorization target of 80% L1 reduction or L1 below 0.05. Evidence is under
`runs/kaggle/selected_runtime_clean_memorization_v8`. Its legacy tiny-summary status is
expectedly `fail` only because that old gate requires exactly 128 updates; the v8 kernel
itself completed and the raw 512-step evidence is valid.

The same clean probe ran for 1024 updates as Kaggle kernel v9. It completed 1024 two-rank
updates with zero skips/non-finites; smoothed L1 improved `0.2534 -> 0.0788` (68.9%) and
reconstruction loss `0.3194 -> 0.1230` (61.5%). Every 64-step bin improved and the last
step reached L1 `0.0780`, but the run did not meet the deliberately strict 80% reduction
or L1-below-0.05 target. These are deterministic clean fixed-training-batch measurements.
The saved image artifact is instead a clean held-out validation sample and must not be
used to diagnose fixed-set memorization. Evidence is under
`runs/kaggle/selected_runtime_clean_memorization_v9`.
The one-off kernel branch was removed after the run; the compact probe config/evidence
contract remains. Kaggle v10 then completed the paired regularized fixed-32 probes with
the same seed, 512-step beta ramp, and 1024 updates. In the post-training clean
`model.eval()`, `z=mu`, beta-zero evaluation, final beta `0.01` produced reconstruction
loss `0.11872`, L1 `0.07748`, SSIM `0.58764`, and unweighted KL `0.47441`; final beta
`0.1` produced `0.14724`, `0.09708`, `0.49837`, and `0.10411`, respectively. Both runs
completed with zero skips/non-finites. Beta `0.1` compresses more but materially degrades
the retained image information, so it is not the default candidate. Evidence is under
`runs/kaggle/selected_runtime_beta_probe_v10`. The original beta-1 run drove KL
effectively to zero.

Latest local verification after the resume/AMP observability addition: the focused
full-run suite passes 216 tests, the dedicated full-kernel preflight passes 217 tests, and
`./scripts/python_quality.sh` passes formatting, Ruff, 687 tests with 1 skip, and
BasedPyright with 0 errors. The repo/workspace preflights and `git diff --check` pass.
Earlier clean-context audits found no launch blocker in
the checkpoint/session/runtime slice. They
confirmed atomic checkpoint publication, the hashed 3000-step checkpoint as the session
commit point, index-only loader continuation, rank/segment-separated stochastic streams,
fixed-25 completion before checkpoint commitment, exact generated-wrapper verification,
beta `0.01`, and the measured Torch `2.13.0+cu130` / CUDA `13.0` stack. The checkpoint
state was also cross-checked against `kaggle/fsq_train_reference.py`: model, optimizer,
scaler, RNG, progress, and best metric are covered; LR/beta progress derives from the
absolute successful-update count. Source commit `81b5017` is on GitHub. Kaggle session 1
version 2 ended `ERROR`: logs show completed boundaries through update 15000 and an AMP
overflow guard failure in the window ending at update 18000. Its downloaded proof,
checkpoint hash, metric prefix, and fixed-25 boundaries verify update 15000 as the commit
point. Session 2 then advanced the committed prefix through update 45000; its exact
checkpoint hash, loadable state, finite successful rows, validation curve, fixed-25
boundaries, and partial manifest are verified locally. Session 3 advances through 60000
with a loadable hash-matched final checkpoint and complete boundaries. Across updates
3000 to 60000, clean L1 improves `0.07733 -> 0.05925` and SSIM `0.5690 -> 0.7336`;
denoising L1 improves `0.08172 -> 0.06236` and SSIM `0.5647 -> 0.7257`. Update-60000
reconstructions preserve tissue/stain morphology without collapse and show expected VAE
smoothing. Rotated-input/rotated-embedding grids, latent tensors, PCA/first-three views,
and error maps are complete; rotation exactness is zero, while the non-equivariant
control's learned representation is correctly not claimed to be equivariant.

## Fresh-agent execution order

1. Keep beta `0.01` locked for the matched baseline/continuous-`SO(2)` comparison; beta
   `0.1` is rejected and no intermediate beta probe is planned. Downstream performance
   will evaluate usefulness, not reopen beta tuning by default.
2. Keep the multi-session implementation deliberately small: let Kaggle close an
   unfinished worker, resume from its latest complete 3000-step checkpoint, and
   concatenate downloaded CSVs locally after completion. Do not build an artificial
   session cap, remote artifact-tree transport, generalized session manager, merge
   service, or cleanup framework.
3. Treat paired-probe removal, index-only resume, per-rank RNG rebasing, atomic
   checkpoint publication, focused tests, full quality, full-kernel preflight,
   repo/workspace preflights, and the post-fix clean-context audits as complete.
4. Treat source commit `81b5017`, GitHub push, guarded API check, Kaggle kernel version 2
   push, and terminal `ERROR` status/log reads as complete. Do not continuously poll.
5. Treat the session-1 output download, update-15000 proof/hash/CSV/fixed-25 verification,
   FSQ comparison, FSQ-aligned AMP runtime/full-output correction, full quality gate, and
   full-kernel preflight as complete. The checkpoint is accepted with one lost physical
   optimizer update under the user's stated tolerance.
6. Treat the exact session-1 checkpoint upload, session-2 path/hash guard, clean commit
   `65112aa`, GitHub push, version-3 launch, terminal cancellation, output download, and
   committed update-45000 verification as complete.
7. Treat the smallest one-off session-3 transport as locally ready at ignored
   `runs/kaggle/session2_resume_dataset`: it contains only dataset metadata and the
   verified update-45000 checkpoint. Final-run metadata, wrapper, and launch guards are
   pinned to private slug `maximusshtefan/eqvae-baseline-session2-step45000`, exact
   Kaggle path, and SHA-256. Do not build a generic session manager.
8. Treat the private checkpoint-dataset upload, source commit `462c9e1` GitHub push,
   guarded final-kernel push, and Kaggle version-4 execution as complete. Remote logs
   show completed boundaries 48000, 51000, 54000, 57000, and 60000, with six isolated
   AMP skips that each recovered on the following attempt. The remote output lists
   `step_060000.pt`, `final.pt`, summaries, manifests, and fixed-25 artifacts.
9. Treat the separate version-4 download, final-checkpoint hash/schema/state validation,
   committed-range concatenation, metric coverage analysis, and update-60000 visual
   inspection as complete. Preserve all three raw downloads; the combined view is derived
   and explicitly non-authoritative. Accept only the known update-14007 physical skip
   under the user's stated tolerance; do not hide or rewrite its legacy telemetry.
10. Treat Spec 0012's radial basis oracle, F01 selection, equal-copy count/init
    refresh, manifest handoff, and Spec 0013's fixed local mechanics/CPU proof
    as complete. Treat private dual-T4 versions 1 and 2 as measured mechanics
    failures, not infrastructure failures, and preserve both tracked summaries.
    The user replaced the isolated 10% gate and selected padded `bmm` plus direct
    assembly. That singular path, its focused local checks, reviews, and guarded
    runner are complete. Accept Kaggle v3 under the user's decision that timing
    CV and one-time compilation duration are diagnostic, not architecture gates.
    Do not rerun or add another arm. Treat Spec 0014's full fixed VAE assembly,
    exact counts, focused verification, and corrected fresh reviews as complete.
11. Treat Spec 0015 readiness, Spec 0016 prelaunch, and SO2 full sessions 1-5
    through update 45000 as complete. Do not rerun them. Preserve each raw
    output separately; the downloaded checkpoint proof and matching checkpoint
    bytes, not Kaggle's unreliable file-list endpoint or an old wrapper, define
    the continuation point.
12. Continue only from the exact update-45000 checkpoint in the session-6
    handoff above. The user accepted retaining session 4's 67/68 and session 5's
    66/68 gate caveats. The locally preflighted transport is pinned to
    `maximshtefan/eqvae-so2-session5-step45000`; its private dataset is version
    1 (ID `11701181`) and session-6 kernel v3 is terminal `ERROR` before
    checkpoint load because Kaggle omitted the exact correct input mount. Do not
    infer a boundary until a later output proof verifies one.

Baseline full training/final-output verification and continuous-`SO(2)` prelaunch
are complete. SO2 full training is committed through update 45000; the active
task is the exact independently operated session-6 checkpoint transport from
the hash-pinned `step_045000.pt`.
