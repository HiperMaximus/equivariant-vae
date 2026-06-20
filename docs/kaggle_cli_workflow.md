# Kaggle CLI Workflow

Status: draft workflow scaffold; synthetic setup-smoke path ready after
permission; synthetic binary timing pretest evidence complete for screening;
capped real-data runtime pretest has a local non-promotable runner/kernel/guard,
upload-simulation proof, and identity/hash/CRC/window plus clean-validation
loader proof plumbing plus a candidate-linked evidence lane implementation;
remote v5 produced two eligible eager bs4 rows and remote v6 completed with
downloaded non-promotable phase-timing plus failed-candidate hash diagnostics
but still only two eligible bs4 rows; remote v7 completed/downloaded and exposed
the repeated failed-candidate exception as `quantile() input tensor is too
large`; remote v8 completed/downloaded, fixed that quantile evidence-plumbing
failure, and produced six capped-pretest-passing eager single-visible-T4 bs4/bs8/bs12 FP32
rows while remaining non-promotable with no selected runtime; compiled rows
remain diagnostic-only until full compile-settle coverage exists; Kaggle source
attachments require a separate confirmation guard
Last updated: 2026-06-20

Kaggle is a remote execution surface, not a Git remote. This repo remains the
source of truth for experiment code, specs, configs, and paper-facing claims.

## Current State

Historical Kaggle notebooks live in:

```text
kaggle/train_runs
kaggle/dataset_generation
kaggle/generate_dataset_Classification_With_Masks
```

They are JSON notebooks kept as historical evidence and behavior-inventory input.
Do not edit them into the new baseline.

The first CLI-managed script-kernel scaffold lives in:

```text
kaggle/kernels/non_eq_vae_debug
```

It now launches only the capped `kaggle_smoke_ready` debug path. That path is
allowed to run at most three train steps and one clean-validation batch, writes
`benchmark/kaggle_smoke.json`, and keeps `full_run_eligible = false`. It is not
runtime selection, convergence evidence, a full benchmark, or a full run.

Important correction from the first remote debug push: the Kaggle CLI script
kernel upload serialized the declared `code_file`, so the sibling
`payload/` directory prepared for `non_eq_vae_debug` was not available remotely.
The first remote version ended in `KernelWorkerStatus.ERROR` with
`ModuleNotFoundError: No module named 'eqvae'` and produced no benchmark
artifact. The real-data smoke launcher has since been migrated locally to
embedded single-file packaging with an upload-simulation import test. A fresh
remote real-data rerun is allowed only when intentionally accepting real dataset
attachment/setup cost; it is not the next step for synthetic/random
training-time benchmarking.

Important correction from the 2026-06-17 remote-control testing pass: the
real-data smoke path is the wrong tool for synthetic/random timing-plumbing
benchmarks because its metadata attaches the 60 GB+
`maximusshtefan/patches-pre-shuffled-ubc-ocean` dataset. It should be used only
when intentionally testing real dataset attachment plus UBC shard resolution.
`scripts/kaggle_kernel.sh push` now requires both `KAGGLE_PUSH_CONFIRMED=1` and
`KAGGLE_FULL_DATASET_CONFIRMED=1` before any metadata with nonempty Kaggle
source attachments can be uploaded. The guarded source fields are
`dataset_sources`, `competition_sources`, `kernel_sources`, and
`model_sources`. The current real-data smoke guard additionally requires the
known patch dataset as the only source attachment. The next training-time
efficiency benchmark should use a separate no-dataset synthetic/random kernel,
not the real-data smoke.

The setup-only script-kernel scaffold lives in:

```text
kaggle/kernels/setup_smoke
```

It attaches no datasets, requests no GPU, uses a generated ignored `run.py` that
embeds a zipped repo payload, generates tiny synthetic UBC-format shards under
the output directory, and writes non-promotable
`benchmark/kaggle_setup_smoke.json`. It validates Kaggle API/script/import/
artifact plumbing only; it is not real-data loader evidence, runtime selection,
or convergence evidence.

Remote setup-smoke v1 was pushed on 2026-06-17 after explicit permission and
completed successfully. The downloaded ignored artifact at
`runs/kaggle/setup_smoke/benchmark/kaggle_setup_smoke.json` records
`status = "smoke_pass"`, `status_scope = "non_promotable_setup_smoke"`,
`benchmark_kind = "synthetic_kaggle_setup_smoke"`, no dataset slug, synthetic
data origin, CPU runtime, `requires_cuda_t4 = false`, 3 train steps, 1
clean-validation batch, 2 deterministic applied corruptions, and clean embedded
payload provenance for commit `3162bec`.

The no-dataset GPU timing scaffold is:

```text
kaggle/kernels/synthetic_timing
```

It is implemented locally at this path, and remote versions 1, 2, 3, and 4
completed on Kaggle with non-promotable `synthetic_timing_pass`. Its contract
is to request T4 GPU runtime, attach no Kaggle datasets or other sources, generate
deterministic UBC-format binary+CSV shards under the Kaggle working output, and
write only non-promotable synthetic timing artifacts. The current default
profile is `synthetic_binary_2gib_histology_like_v1`: 10,912 total
`3x256x256` CHW `uint8` patches, split 5,456 train / 5,456 validation,
2,145,386,496 payload bytes before CSV/artifacts, about 1.998 GiB. This keeps
non-wrapping 30-batch throughput rows eligible through global batch 128. Remote
v1 used the earlier 0.81 GB profile; remote v2 refreshed the evidence with this
2 GiB-scale default; remote v3 added per-rank DDP device proof and corrected
`drop_last = false` projection fields; remote v4 is the current
5-warmup/25-measured repeat-shortlist evidence at
`runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark`.

The generated synthetic root must mirror the real shard contract exactly enough
to exercise the same loader path: write
`dataset/ubc_train_shuffled.bin`, `dataset/ubc_train_shuffled.csv`,
`dataset/ubc_ocean_valid.bin`, and `dataset/ubc_ocean_valid.csv`; use the same
64-byte header, CHW `uint8` payload order, CRC32, and CSV metadata semantics;
omit `idx` from the train CSV and include `idx` in the validation CSV. Timed
reads must resolve this explicit `/kaggle/working/...` root through
`resolve_patch_data_paths` and use the active `PatchTensorDataset` /
`PatchTrainingDataset` code, not a synthetic-only loader.

This pretest must attempt both `single_visible_t4` and `dual_t4_ddp` so batch
size and VRAM differences are visible. Compare rows by feasible global
throughput and projected epoch time, not by equal per-device batch size. The
ranked non-wrapping eligibility budget is
`global_batch_size * non_wrapping_eligibility_steps <= split_patch_count`, with
`non_wrapping_eligibility_steps = 30` for the default profile. An initial
shorter fit pass may still run larger rows, but rows that exceed the 30-step
budget are fit/VRAM probes only. The output may recommend rows to carry into
the real-data benchmark, but must not write `benchmark/selected_runtime.json`
or claim final
runtime selection.

The real-data benchmark surface is intentionally separate from the capped smoke
and from synthetic timing:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest
```

Its config/schema contract lives in
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` and is
`real_data_runtime_pretest_contract_ready`. The local runner/kernel/guard
implementation is non-promotable: it attaches only
`maximusshtefan/patches-pre-shuffled-ubc-ocean`, uses the fixed 8,192-train /
2,048-validation spread windows recorded in the config, writes blocked claims
and pretest recommendations, and must not write
`benchmark/selected_runtime.json`. It uses the synthetic-v4 rows only as parent
provenance and adds sentinel rows. The local proof lane records real-data/local
file identity, hashes, CRCs, row counts, locked train/validation windows, split
WSI/holdout overlap contracts, and a clean validation loader/collate/
normalization proof. Tiny fixture roots can only produce `local_pass`;
canonical real `pass` requires the exact dataset slug, 300000/30000 rows,
322/39 WSIs, zero train/validation and masked-holdout overlap, and the locked
8,192/2,048 spread windows. The candidate-linked evidence lane can record
measured `model_forward` compile-settle/Dynamo counters, run a real
`torchrun --standalone` dual-rank DDP launch probe when two T4s are visible,
measure fixed-window dataloader throughput per accelerator/batch candidate, and
attach same-batch eager-reference numerical checks, corruption checks, and
gate-health evidence back to the exact runtime row identity. Tiny fixture roots
still produce only local mechanics proof; remote canonical eager-row
eligibility requires the real-data proof, real DDP launch proof, row-matching
dataloader/numerical/corruption/gate-health status, and zero graph-break/
recompile counts. Compiled `model_forward` rows must remain ineligible/
diagnostic until full compile-settle coverage includes clean validation, DDP
rank paths, final partial batches, and mask cardinalities 0/1/many/all. Remote
pushing requires explicit user permission plus:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest
```

Spec 0001 runtime benchmarking requires two accelerator modes:

- `single_visible_t4`: run on one visible GPU with `world_size = 1`;
- `dual_t4_ddp`: run on two T4 GPUs with `world_size = 2`.

The preferred implementation is one dual-T4 benchmark kernel where
`single_visible_t4` restricts visibility to the first GPU and `dual_t4_ddp`
launches two ranks with `torchrun --standalone --nproc_per_node=2` or an
equivalent self-spawn launcher.

Verified Kaggle accelerator metadata:

- on 2026-06-11, `kaggle kernels pull maximusshtefan/non-eq-vae -m` downloaded
  metadata for the existing notebook that the Kaggle UI showed as GPU T4 x2;
- the pulled `kernel-metadata.json` has `"machine_shape": "NvidiaTeslaT4"`,
  `"enable_gpu": true`, and `"enable_tpu": false`;
- Kaggle CLI 2.2.1 accepts `--accelerator ACC` and passes that string directly
  as `request.machine_shape`, so benchmark metadata/tooling should use
  `NvidiaTeslaT4` rather than inventing a separate `T4x2` string.

Because the metadata field does not encode the count of visible T4 devices, the
benchmark must verify the actual allocation at runtime. `dual_t4_ddp` rows must
record `cuda_device_count == 2`, two T4 device names, `world_size == 2`, and
`nproc_per_node == 2`; otherwise the row fails with
`failure_kind = "wrong_accelerator"`. `single_visible_t4` rows may use the same
Kaggle machine shape but must mask visibility to one GPU and record
`visible_device_count == 1`.

The behavior inventory now lives at:

```text
docs/behavior_inventory_kaggle.md
```

On 2026-06-06, `./scripts/kaggle_kernel.sh check` passed on this laptop with
Kaggle CLI 2.2.1. Authentication is still a user-local secret and should be
rechecked before remote reads or writes.

On 2026-06-11, the read-only API preflight command:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
```

confirmed:

- OAuth access-token generation works without printing the token;
- `kernels list`, `kernels status`, and `kernels logs` work for
  `maximusshtefan/non-eq-vae`;
- `datasets files` works for
  `maximusshtefan/patches-pre-shuffled-ubc-ocean`;
- `kaggle quota -v` fails with Kaggle's authentication-required message even
  though OAuth token generation works;
- `kaggle kernels files maximusshtefan/non-eq-vae -v` also fails with the same
  authentication-required message.

Therefore the benchmark workflow must not rely on the CLI quota endpoint as the
only gate. Before a remote benchmark push, run the API preflight, check GPU
quota/availability in the Kaggle web UI if the quota endpoint still warns, and
let the benchmark itself fail rows with `failure_kind = "wrong_accelerator"` if
Kaggle does not allocate two visible T4 devices for `dual_t4_ddp`.

The user visually confirmed the Kaggle web UI quota on 2026-06-11: phone
verification is complete, identity verification is not complete, and Kaggle GPU
quota shows `00:07 / 30 hrs` used. Identity verification is not currently a
benchmark blocker as long as the UI continues to expose GPU quota and notebook
GPU selection.

## Local Commands

Validate the local scaffold:

```bash
./scripts/kaggle_kernel.sh validate
```

After spec 0001 implementation creates repo code/configs, build the local
real-data payload before local validation. This legacy sibling payload is not
sufficient for remote execution until the real-data launcher is migrated to a
proved upload mechanism:

```bash
./scripts/kaggle_kernel.sh build
```

The generated `kaggle/kernels/*/payload/` directory is ignored and must be
rebuilt from source before remote pushes. Payload metadata includes both
`pyproject.toml` and `uv.lock`; spec 0001 kernels must not resolve or install
dependencies on Kaggle unless a later spec explicitly changes that rule.

Build the synthetic no-dataset setup smoke as a single generated upload file:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke
```

That command writes ignored `kaggle/kernels/setup_smoke/run.py` and verifies its
embedded zip and manifest. The setup push guard decodes the generated file again
and rejects stale payloads before any remote write.

For a setup-only remote smoke, the intended sequence is:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/setup_smoke
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-setup
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-setup
```

Run those remote commands only after explicit user permission. A successful
setup smoke should produce `benchmark/kaggle_setup_smoke.json` with
`status = "smoke_pass"`, `status_scope = "non_promotable_setup_smoke"`,
`benchmark_kind = "synthetic_kaggle_setup_smoke"`, no dataset slug, no Kaggle
input mount origin, `requires_cuda_t4 = false`, a payload manifest, nonzero
optimizer updates, and at least one deterministic applied corruption.

For a fresh capped real-data smoke after clean commit/rebuild, the intended
remote sequence is:

```bash
./scripts/kaggle_kernel.sh build
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
```

Run those remote commands only after explicit user permission that includes
accepting Kaggle dataset attachment/setup cost. A successful smoke should
produce `benchmark/kaggle_smoke.json` with
`status = "smoke_pass"`, `status_scope = "non_promotable_debug"`,
`full_run_eligible = false`, at least one applied corruption, nonzero
input-target delta, nonzero optimizer updates, visible T4 CUDA runtime for the
real-data path, payload-manifest provenance, and explicit
`data_integrity_status`. The first remote version is already `ERROR` from the
missing source package and produced no smoke evidence.

Use the real-data smoke only when intentionally testing Kaggle dataset
attachment plus UBC shard resolution. The `patches-pre-shuffled-ubc-ocean`
source is larger than 60 GB, so Kaggle may spend a long time preparing the
environment before a capped script starts. For setup-only tests of the API,
script-kernel payload, imports, artifact writing, or synthetic/random
training-time plumbing, use a no-dataset kernel with empty `dataset_sources`.
The existing setup smoke has a distinct status/source and must never be promoted
to real-data benchmark evidence.

After real-data runtime pretest v2 completed with `data_root_unavailable`, the
pretest resolver records full `real_data_proof.data_root_diagnostics` and a
short stderr JSON probe for Kaggle input resolution. Auto discovery may only
promote complete shard roots under the expected
`maximusshtefan/patches-pre-shuffled-ubc-ocean` mount family; unrelated complete
shard roots under `/kaggle/input` are diagnostics-only and must not be selected.
Remote v3 confirmed the expected Kaggle shard root and then exposed a separate
payload issue: the embedded single-file kernel must include
`docs/data/ubc_ocean_masked_holdout_ids.csv` for split/holdout-overlap proof.
Remote v4 included that file and passed the canonical real-data identity,
row-count, WSI/holdout overlap, CRC, locked-window, and clean-validation-loader
proof lane. Remote v5 completed as a non-promotable candidate-evidence run and
downloaded artifacts under `runs/kaggle/real_data_runtime_pretest_v5`: two eager
single-T4 bs4 FP32 rows became row-eligible, real-data/DDP/dataloader/gate lanes
passed, no `benchmark/selected_runtime.json` was written, eager bs8/bs12 still
need numerical/corruption evidence coverage, eager bs32 rows record
`runtime_OutOfMemoryError`, and compiled rows remained diagnostic/ineligible as
intended. The v6
follow-up prioritizes eager FP32 single-T4 train-step evidence by smaller batch
size before compiled diagnostic rows, clears CUDA cache between candidate
evidence attempts, records failed candidate evidence with deterministic failure
hashes, mirrors candidate/failed evidence counts into `runtime_proof.json`, and
keeps the full failed-evidence list in the paired numerical and corruption proof
objects. The phase-timing artifact must be present in both the payload artifact
allow-list and the launcher validation allow-list; the local generated-launcher
full simulation covers this exact artifact set. Commit `47437a0` was rebuilt,
validated, and pushed as Kaggle version 6 after explicit approval; v6 completed
and downloaded artifacts live under `runs/kaggle/real_data_runtime_pretest_v6`.
Inspection found no new runtime selection: two eager single-T4 bs4 FP32 rows
remain eligible, bs8/bs12 candidate evidence failed with hash-only
`candidate_train_step_RuntimeError` diagnostics, eager bs32 rows record
`runtime_OutOfMemoryError`, compiled rows remain diagnostic/ineligible, and no
`benchmark/selected_runtime.json` was written. The local v7 diagnostics
follow-up added bounded `failure_message_excerpt` fields. It was pushed as
Kaggle version 7 after explicit approval and the required guards, completed,
and downloaded to `runs/kaggle/real_data_runtime_pretest_v7`. Inspection found
no new runtime selection: two eager single-T4 bs4 FP32 rows remain eligible,
bs8/bs12 and compiled candidate evidence are now diagnosed as failing in
gate-health quantile telemetry with `quantile() input tensor is too large`,
eager bs32 rows record `runtime_OutOfMemoryError`, compiled rows remain
diagnostic/ineligible, and no `benchmark/selected_runtime.json` was written.
The local v8 slice implemented a deterministic bounded/sampled gate-health
quantile path for `gate_p01/gate_p50/gate_p99`, preserved exact full-tensor
finite/saturation/worst-channel/dead-channel pass/fail checks, and prevented
lane-level gate-health success from covering rows without matching
row-specific evidence. It was committed as `614cd95`, rebuilt and validated from
a clean source state, pushed as Kaggle version 8 after explicit approval and the
required guards, completed, and downloaded to
`runs/kaggle/real_data_runtime_pretest_v8`. Inspection found the v7 quantile
failure is fixed: paired-numerical and corruption failed-candidate evidence
counts are both zero. Six eager single-visible-T4 FP32 rows pass in the capped
pretest: bs4/bs8/bs12 crossed with `branchless_all` and
`indexed_masked`. The run still wrote no `benchmark/selected_runtime.json`,
still has `runtime_proof.status = pretest_incomplete`, and remains
non-promotable. Eager bs32 remains `runtime_OutOfMemoryError`, dual-T4
train-step measurement remains pending, and compiled rows remain
diagnostic/ineligible until full compile-settle evidence exists.

The next selected-runtime benchmark/debug slice is
`v8_shortlist_eager_amp_then_dual_gate` in
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json`. It treats v8 only
as shortlist provenance. A separate runtime-selection benchmark must revalidate
or write its own linked proofs, confirm the eager single-visible-T4 bs8/bs12
FP32 `compile_none` branchless/indexed rows with bs4 as fallback, run AMP
follow-up only on confirmed eager rows, add the blocking real dual-T4 train-step
timing gate, and write `benchmark/selected_runtime.json` only after its own full
linked proof passes.

Local `build`/`validate` may verify the generated real-data pretest payload
against the current dirty worktree so agents can validate local patches before
commit. Remote push safety is stricter: the real-data pretest push guard still
rejects a generated manifest with `git_dirty = true`, so future pushes require a
clean committed source state and a fresh rebuild/validate first. The exact
guarded sequence used for v8, and required again for any rerun or successor
remote slice, is:

```bash
git status --short
./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest
./scripts/kaggle_kernel.sh validate kaggle/kernels/real_data_runtime_pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
# Confirm Kaggle web UI GPU quota if the CLI quota endpoint still warns.
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v8
```

Do not run any `KAGGLE_REMOTE_CONFIRMED=1` or `KAGGLE_PUSH_CONFIRMED=1`
command without explicit permission.

### Remote Duration And Polling Memory

Do not poll long-running Kaggle kernels continuously. Remote reads still require
explicit approval and `KAGGLE_REMOTE_CONFIRMED=1`, and large source-attached
runs can spend many minutes preparing the environment before the script emits
useful artifacts.

Default polling cadence:

- after a push, one immediate `status` check is allowed to confirm that Kaggle
  accepted the version and moved it into a worker state;
- for real-data kernels that attach `patches-pre-shuffled-ubc-ocean`, wait at
  least 30 minutes before the next status check, then poll at 30-minute or
  slower intervals until a terminal state appears;
- do not repeat direct log reads when logs are empty for a running worker;
- once a run completes, download artifacts once and record the observed duration
  before deciding future cadence.

Record timing memory in `CURRENT.md` and, when stable, in this section. Capture:
push/version acceptance time, first observed `RUNNING` time, terminal status
time, output-download time, and artifact phase timings such as data-root
resolution, clean-validation proof, DDP launch proof, dataloader throughput,
numerical checks, corruption checks, and gate-health work if the artifact
contains those fields.

Observed v7 timing memory, 2026-06-20: Kaggle accepted version 7 at
`https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`;
the immediate guarded status read returned `KernelWorkerStatus.RUNNING` at
`2026-06-19T23:38:51-05:00`, the next guarded poll returned
`KernelWorkerStatus.COMPLETE` at `2026-06-20T02:21:15-05:00`, and outputs were
downloaded to `runs/kaggle/real_data_runtime_pretest_v7` at
`2026-06-20T09:19:52-05:00`. Artifact phase timings recorded
`2026-06-20T05:02:18Z` to `2026-06-20T05:40:51Z` with 71 passing phases.

The real-data runtime pretest runner writes coarse JSON-line phase events to
stderr and, for versions built after this logging slice, writes
`benchmark/phase_timings.json` plus matching `phase_timings` objects in
`runtime_proof.json` and `real_data_runtime_pretest_manifest.json`. Use those
durations to update future polling cadence instead of guessing from status
checks.

Current duration notes:

- Real-data runtime pretest v5, 2026-06-19: Kaggle accepted version 5, kept
  reporting `KernelWorkerStatus.RUNNING` throughout the initial monitoring
  window, and later reached `KernelWorkerStatus.COMPLETE`. Downloaded artifacts
  live under `runs/kaggle/real_data_runtime_pretest_v5`. The Kaggle log reports
  data-root probing at about 7.44 seconds and notebook result conversion around
  2355 seconds, so use roughly 40 minutes as the first observed duration for
  this capped source-attached pretest. Version 5 predates
  `benchmark/phase_timings.json`; future reruns should use the phase timings
  artifact for more exact cadence and should inspect the runtime-proof evidence
  counters before assuming a candidate lane silently skipped work.
- Real-data runtime pretest v6, 2026-06-19: Kaggle accepted version 6 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`.
  The preflight again passed OAuth, kernel list/status/logs, and patch-dataset
  file listing while warning on quota and kernels-file introspection. The first
  status read reported `KernelWorkerStatus.RUNNING` by
  2026-06-19T16:37:07-05:00. A later approved status read after the required
  wait reported `KernelWorkerStatus.COMPLETE`, and artifacts were downloaded to
  `runs/kaggle/real_data_runtime_pretest_v6`. The phase-timing artifact records
  `started_at_utc = 2026-06-19T21:36:15Z`,
  `finished_at_utc = 2026-06-19T22:15:40Z`, and 71 passing phase records.
  Longest phases were stage1 runtime rows at about 1185.75s, linked evidence
  payload at about 592.19s, real-data identity/clean-path proof at about
  586.85s, and linked train-step evidence at about 573.67s.

For the synthetic binary timing pretest, the remote sequence remains permission
gated:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/synthetic_timing
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/synthetic_timing
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status <synthetic-kernel-id>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output <synthetic-kernel-id> runs/kaggle/synthetic_timing
```

Until `api-check` grows a no-dataset kernel mode, the command above is only a
read-only auth/status/quota preflight and still lists the real patch dataset.
It must not be interpreted as synthetic dataset attachment.

Do not set `KAGGLE_FULL_DATASET_CONFIRMED=1` for this no-dataset kernel. The
push guard must instead prove all Kaggle source lists are empty, GPU/T4 metadata
is present, the generated payload is fresh, and the launcher cannot write
promotable runtime-selection artifacts.

Check whether the Kaggle CLI is installed and whether local metadata is valid:

```bash
./scripts/kaggle_kernel.sh check
```

After explicit user permission for remote reads, run the read-only API
preflight before remote benchmark pushes:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check
```

Push a no-source script kernel, such as setup smoke or synthetic timing, only
after explicit user permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/setup_smoke
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/synthetic_timing
```

Push a source-attached real-data kernel only after explicit user permission and
after intentionally accepting the real dataset attachment/setup cost:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest
```

Check remote status after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
```

Download outputs into ignored local run artifacts after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v6
```

Pulling from Kaggle can overwrite local files and requires explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 KAGGLE_PULL_CONFIRMED=1 ./scripts/kaggle_kernel.sh pull
```

## Credentials

Kaggle credentials are local secrets. Do not store, print, or commit them.

The official Kaggle API supports local CLI authentication and the standard local
token file. Agents must ask before running network commands or remote writes.

## Dataset Sources

Attach Kaggle datasets through `kernel-metadata.json`, not by hard-coding UI
display names in the script.

Use exact dataset slugs, for example:

```json
"dataset_sources": ["owner/dataset-slug"]
```

The current scaffold uses the confirmed pre-shuffled patch dataset:

```json
"dataset_sources": ["maximusshtefan/patches-pre-shuffled-ubc-ocean"]
```

Other confirmed historical slugs are recorded in
`docs/behavior_inventory_kaggle.md`. Do not attach
`maximusshtefan/non-eq-vae-output` to spec 0001 or any new normal VAE baseline.
A future historical-reproduction spec would need to opt into that source
explicitly.

The pre-shuffled patch dataset is the confirmed train/validation patch source.
It contains `ubc_train_shuffled.*` and `ubc_ocean_valid.*`, but no held-out test
shard. Final evaluation needs a separate sealed test dataset/source from the
UBC-OCEAN WSIs with supplemental masks. Those masks are non-exhaustive and should
not be interpreted as full-WSI negative/positive coverage.

For the current real-data smoke kernel, the push wrapper refuses remote writes
while `dataset_sources` is empty, while the bundled payload is missing or stale,
while the dataset slug differs from
`maximusshtefan/patches-pre-shuffled-ubc-ocean`, while any of
`competition_sources`, `kernel_sources`, or `model_sources` is nonempty, or
while spec 0001 and the spec index are not marked with the appropriate
readiness label. In addition, any push whose metadata has any nonempty Kaggle
source list fails unless `KAGGLE_FULL_DATASET_CONFIRMED=1` is set for that one
command. The synthetic setup-smoke guard is a separate branch: it requires empty
source lists, no GPU, no internet, a generated embedded payload, and
setup-specific readiness docs, so this real-data dataset requirement is not
weakened.

For spec 0001 benchmark kernels, the wrapper or metadata validation must require
`machine_shape == "NvidiaTeslaT4"` and the single-visible versus dual-DDP launch
mode recorded above.

For the synthetic timing kernel, metadata validation must require the same T4
machine shape but empty source lists. `benchmark/synthetic_timing_runtime_proof.json`
must record visible CUDA device count, T4 device names, `world_size`,
`nproc_per_node`, per-rank device assignment, and whether `dual_t4_ddp` was
measured or failed with `wrong_accelerator`/`skipped_unsupported`.

## GitHub Linking

Kaggle's web UI can show a notebook as linked from GitHub, but that is not the
workflow here. For agentic work, the repo should generate or own the script
kernel folder, and the Kaggle API should upload that folder.

If someone edits a kernel in the Kaggle UI, pull it locally, inspect the diff,
and reconcile it into the repo. Do not let UI edits become the source of truth.

## Official References

- Kaggle API README: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- Kaggle kernel commands: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md
- Kaggle kernel metadata: https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
