# Kaggle CLI Workflow

Status: draft workflow scaffold; runtime-selection v5 is the selected fallback
runtime (`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`,
`27.381321` samples/sec, about 30.4 projected hours for 10 epochs);
runtime-selection v6 tested relaxed scalar-gate AMP and kept v5 because it was
slower. Spec 0008 selected-runtime debug/tiny v5 completed on Kaggle,
downloaded to `runs/kaggle/selected_runtime_debug_v5`, and passed strict
`eqvae.cli.selected_runtime_gate --verify-output` with canonical real fixed-32
selector generation, selected-runtime plan application, checkpoint/resume, gate
health, artifact manifest, zero tiny AMP skips, zero tiny nonfinite rows, and no
launch blockers. The next work is not another debug/tiny push. Spec 0009 now
has a guarded first full selected-runtime training workflow. The first full
kernel push was explicitly approved and accepted by Kaggle as
`maximusshtefan/eqvae-selected-runtime-full` version 1 on 2026-06-29; it has
not yet been status-checked, downloaded, or verified.

Current full-run surface: `scripts/kaggle_kernel.sh` has
`preflight-selected-runtime-runner`, `preflight-selected-runtime-debug`,
`preflight-selected-runtime-full`, `status-selected-runtime-full`, and
`output-selected-runtime-full`. The full launcher is the dedicated
`kaggle/kernels/selected_runtime_full` kernel. Do not reuse
`kaggle/kernels/selected_runtime_debug`, which remains bounded to
debug/resume/tiny proof steps and writes non-promotable artifacts. There is no
`push-selected-runtime-full` action; full pushes use
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_full`.
The next remote actions are status/output checks, each requiring exact approval
and `KAGGLE_REMOTE_CONFIRMED=1`. Kaggle source attachments require a separate
confirmation guard.
Last updated: 2026-06-29

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
Do not edit them into the new baseline. `kaggle/train_runs` is the successful
working FSQ autoencoder training notebook/artifact; use it as reference for the
broad macro-architecture and runtime tactics, while excluding FSQ
quantization/codebooks/rounding/discrete latents from the continuous `SO(2)`
route.

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

The selected-runtime benchmark slice is separate from both synthetic timing and
the capped real-data pretest. Local fail-closed proof plumbing lives in:

```bash
python -m eqvae.cli.runtime_selection_benchmark \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --output-dir /tmp/eqvae-runtime-selection-local \
  --v8-artifact-dir runs/kaggle/real_data_runtime_pretest_v8
```

That local invocation records v8 artifact hashes as shortlist-only provenance
and writes its own runtime proof/matrix/linked safety/model-count artifact
graph, but it must remain blocked and write no
`benchmark/selected_runtime.json` without real dual-T4 DDP train-step timing
evidence.

The Kaggle executor and single-file script kernel now live in:

```bash
python -m eqvae.cli.runtime_selection_executor \
  --config configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json \
  --output-dir /tmp/eqvae-runtime-selection-executor \
  --v8-artifact-dir runs/kaggle/real_data_runtime_pretest_v8

./scripts/kaggle_kernel.sh build kaggle/kernels/runtime_selection
./scripts/kaggle_kernel.sh validate kaggle/kernels/runtime_selection
```

The generated runtime-selection kernel embeds only the required v8 provenance
files under `runs/kaggle/real_data_runtime_pretest_v8`; those files remain
shortlist/provenance-only and are never promoted as selected-runtime proof. The
executor must still prove two visible T4s, `world_size = 2`,
`nproc_per_node = 2`, per-rank device assignment, child-process
`torchrun --standalone --nproc_per_node=2` command proof, emitted bs4/bs8/bs12
FP32 eager dual rows, train and validation dataloader rank coverage, scoped
numerical/corruption rows, gate-health evidence bound to candidate row ids, a
hash-linked `benchmark/stain_corruptor_qa.json`, and global throughput
projection before selected-runtime writing is allowed. The strict writer now
also enforces 25 measured dataloader batches with wait/throughput thresholds,
three fixed numerical batch indices, train plus validation clean-RNG corruption
rows, per-candidate gate-health rows, candidate-scoped stain QA, and exact
embedded v8 payload membership. Missing, failed, or skipped dual timing or
linked evidence refuses `benchmark/selected_runtime.json`.

The user approved the selected-runtime Kaggle push/status/output on 2026-06-20.
Runtime-selection v1 reached `KernelWorkerStatus.ERROR` after writing benchmark
artifacts. Inspection confirmed the dual-T4 DDP timing gate passed with two
visible T4s, `world_size = 2`, `nproc_per_node = 2`, per-rank device
assignment, child-process launch proof, emitted bs4/bs8/bs12 FP32 eager dual
rows, and global throughput projection. The strict writer refused
`benchmark/selected_runtime.json` because linked single-visible proof rows were
false-negative blocked by gate-health eligibility normalization and the train
corruption clean-validation RNG check; the wrapper then rejected
`model_inventory.csv` as an unexpected benchmark artifact. The local v2 fix
accepts `model_inventory.csv`, normalizes `local_pass` gate rows before
eligibility is computed while keeping failed non-gate rows ineligible, and
requires `clean_validation_rng_advanced = false` only on validation corruption
rows. Runtime-selection v2 completed and downloaded to
`runs/kaggle/runtime_selection_v2`; it fixed those v1 blockers and proved the
dual gate again, but still refused `benchmark/selected_runtime.json` because
gate-health rows were missing for the three single-visible `indexed_masked`
pass rows. The local v3 fix expands branchless single-visible gate-health rows
to same-shape indexed candidate ids only after the indexed runtime row has
already passed linked evidence. Runtime-selection v3 completed and downloaded
to `runs/kaggle/runtime_selection_v3`; `runtime_proof.status = pass`,
`selection_ready = true`, `selected_runtime_written = true`, and the selected
runtime artifact is
`runs/kaggle/runtime_selection_v3/benchmark/selected_runtime.json`.

Selected runtime: `dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`;
dual T4 DDP, `world_size = 2`, `nproc_per_node = 2`, per-device batch size 12,
global batch size 24, FP32 eager/no compile, `indexed_masked` corruption,
`samples_sec = 14.035497`, estimated epoch time about 356.24 minutes, and
projected 10-epoch wall time about 59.37 hours.

Treat this as the proof-clean safety baseline. Before launching a 60h+ training
run, run an efficiency-selection follow-up that can replace it if AMP/FP16,
stable `torch.compile`, channels-last layout, cuDNN benchmark/non-deterministic
kernel selection, DDP `static_graph`/`gradient_as_bucket_view`, optimizer/
zero-grad fast paths, or Kaggle-supported TF32/matmul precision knobs are
materially faster. The project accepts lost bitwise determinism and small
numerical drift for this speedup; catastrophic failures such as non-finite
loss/gradients, repeated AMP skips, DDP instability, broken checkpoint/resume,
broken artifacts, gate-health collapse, or clearly invalid metrics still block
selection. That first follow-up was implemented as
`selected_runtime_v3_efficiency_followup` with policy-bound rows and linked
proofs; runtime-selection v5 now supersedes it as the fallback, and the local
successor config tested the compact `amp_fp16_scalar_gate_relaxed` comparison
in runtime-selection v6. v6 downloaded and replayed locally, wrote no selected
runtime, and kept v5 before the later debug/resume/tiny-overfit launch gate.

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

On 2026-06-29, the selected-runtime debug/tiny v5 push attempt found another
Kaggle CLI auth edge: OAuth token generation still worked, but ordinary
`kaggle kernels ...` commands reused a stale cached access token and failed
before upload. The selected-runtime slug was correct. `scripts/kaggle_kernel.sh`
now routes authenticated Kaggle reads/writes through
`scripts/kaggle_oauth_exec.py` when `~/.kaggle/credentials.json` exists. The
helper uses the installed Kaggle SDK to generate a fresh short-lived OAuth
token, passes it to the child CLI through a temporary 0600 token file, and
deletes that file when the child exits. This avoids shell token substitution
and token printing. Set `KAGGLE_DISABLE_FRESH_OAUTH=1` only when intentionally
debugging the raw Kaggle CLI auth path.

`api-check` now accepts an optional kernel directory. For the selected-runtime
debug/tiny gate, use:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check kaggle/kernels/selected_runtime_debug
```

That selected-runtime preflight passed on 2026-06-29 through the fresh-token
wrapper with the known quota warning and `kernels files` working. A follow-up
adversarial review removed the old raw `kaggle auth print-access-token` probe
from `api-check`; the command now reports
`ok: fresh OAuth wrapper selected for authenticated Kaggle calls` and proves
auth through wrapped `kernels`/`datasets` endpoint calls.

After fresh explicit approval for the narrow retry, the selected-runtime
debug/tiny v5 push passed the guarded preflight and Kaggle accepted version 5 on
2026-06-29. The immediate guarded status read at `2026-06-29 03:27 -0500`
returned `KernelWorkerStatus.RUNNING`, and the guarded follow-up status read
returned `KernelWorkerStatus.COMPLETE`. Outputs were downloaded to
`runs/kaggle/selected_runtime_debug_v5`; strict `--verify-output` passed with
canonical real fixed-32 selector generation, all selected-runtime debug gate
components passing, no remaining launch blockers, zero tiny AMP skips, zero
tiny nonfinite rows, and 256 nested tiny metric rows across ranks 0/1. The
first full real selected-runtime run is now the next candidate action, with
fresh explicit approval required.

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

The completed selected-runtime benchmark slice
`v8_shortlist_eager_amp_then_dual_gate` in
`configs/spec0001/non_eq_vae_kaggle_runtime_benchmark.json` treated v8 only as
shortlist provenance. The guarded executor then revalidated linked proofs,
confirmed the eager single-visible-T4 candidates, ran the blocking real dual-T4
train-step timing gate, and wrote selected-runtime artifacts only after its own
full linked proof passed. The dual-T4 gate had to record two visible T4s,
`world_size = 2`, `nproc_per_node = 2`, per-rank device assignment,
child-process launch command proof, train and validation dataloader rank
coverage, scoped numerical/corruption rows, gate-health evidence bound to
candidate row ids, a hash-linked `benchmark/stain_corruptor_qa.json`, and
global throughput projection.
Runtime-selection v1 showed the dual gate can pass on Kaggle but selected
runtime writing remained blocked by linked-proof false negatives and wrapper
allow-list drift. The local v2 patch fixes those specific blockers while
preserving the refusal rule for missing, failed, or skipped linked proof.
Runtime-selection v2 then completed without a wrapper error but still refused
selection because single-visible indexed-mask pass rows lacked gate-health rows.
The local v3 patch binds those rows from the branchless gate-health reference
only after the indexed runtime row has already passed linked evidence.
Runtime-selection v3 completed and wrote the selected-runtime artifact:
`runs/kaggle/runtime_selection_v3/benchmark/selected_runtime.json`.
The local selected-runtime efficiency follow-up is implemented after v3. After
adversarial subagent review and explicit approval for the efficiency benchmark
only, local commit `753c9db` was created, the runtime-selection kernel was
rebuilt/validated from the clean commit, and Kaggle accepted version 4. The one
guarded status read returned `KernelWorkerStatus.RUNNING`; on resume v4
completed and outputs were downloaded to `runs/kaggle/runtime_selection_v4`.
Version 4 failed closed with no `benchmark/selected_runtime.json`: the fastest
otherwise clean AMP conservative row projected around 33.0 hours for 10 epochs,
but writer policy false negatives treated small selected-row numerical drift
and nonselected-row proof failures as global blockers. Local commit `fc5227d`
repairs that writer policy and local replay of v4 artifacts passes. Kaggle
accepted runtime-selection version 5 from the clean rebuilt kernel; the one
guarded status read returned `KernelWorkerStatus.RUNNING` at 2026-06-21
06:14:37 -05. On the next approved status read, v5 was
`KernelWorkerStatus.COMPLETE`; outputs were downloaded to
`runs/kaggle/runtime_selection_v5`. The proof passed and selected
`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
with `runtime_policy_id = amp_fp16_conservative`, `samples_sec = 27.381321`,
estimated 10-epoch wall time `109563.740875` seconds, zero AMP skips, no OOM,
gate-health pass, and strict local replay pass under current `main`. The
2026-06-21 pre-long-run preference for a compact broader/non-conservative AMP
follow-up, including `amp_scalar_gate_relaxed` or the closest implemented
less-conservative AMP policy, was satisfied by runtime-selection v6. That
comparison downloaded to
`runs/kaggle/runtime_selection_v6`, replayed locally, and did not write
`benchmark/selected_runtime.json`: the relaxed row passed runtime/gate-health
with zero AMP skips but reached only `25.288828` samples/sec versus v5 at
`27.381321`. Keep v5 as the fallback.

Local-first rule for runtime-selection pushes: before any future
runtime-selection remote push or approval request, run the cheap semantic local
preflight that can catch writer-policy, artifact-shape, generated-wrapper, and
downloaded-artifact replay errors:

```bash
./scripts/kaggle_kernel.sh preflight-runtime-selection
```

That command builds and validates the generated runtime-selection kernel, runs
the runtime-selection writer suite, runs the generated-wrapper import and
fail-closed simulations, and replays downloaded v5 artifacts when they are
present locally. Do not rely on Kaggle exit status alone: the executor can
legitimately return exit code 0 while writing fail-closed proof artifacts, so
local and remote checks must inspect `benchmark/runtime_proof.json` and
`benchmark/selected_runtime.json`.

Local-first rule for selected-runtime debug/tiny pushes: before any remote write
or approval request for the real debug/resume/artifact/tiny-overfit gate, run:

```bash
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
```

That command builds and validates `kaggle/kernels/selected_runtime_debug`, runs
the selected-runtime gate tests, runs the generated-wrapper import-only
simulation, and runs the full local fail-closed artifact simulation. Passing it
means the single-file wrapper can transport v5 and preserve the gate artifact
contract. It does not mean the real UBC proof passed. The push guard remains
stricter and must reject remote writes until the embedded payload no longer has
the Spec 0008 local generator/readiness checks pass in `remote_generate` mode,
exact real-dataset metadata is attached, and the remote kernel is configured to
generate and validate exactly 32 canonical real train selectors from the Kaggle
train shard before training.

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

Agent handoff rule: if a Kaggle push or status read shows the kernel is still
`RUNNING` and the run is likely to take more than about 5 minutes, do not keep
waiting in the same assistant turn. Give the user a concrete local time to
prompt with `continue`, then stop. Resume only when the user prompts again or
explicitly asks for another poll. This avoids burning context/tokens on idle
remote waits.

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
- Runtime-selection v3, 2026-06-20/21: Kaggle accepted version 3 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection`; the
  immediate guarded status read was `KernelWorkerStatus.RUNNING`, so the agent
  stopped waiting and gave the user a concrete prompt time. On resume the
  guarded status read was `KernelWorkerStatus.COMPLETE`, and artifacts were
  downloaded to `runs/kaggle/runtime_selection_v3`. The log reports linked
  train-step evidence at about 183.29s, linked dataloader throughput at about
  5.80s, linked numerical/corruption/gate-health phases under 0.01s each, and
  notebook conversion around 1452.74s. For similar selected-runtime kernels, do
  one immediate status check after push and then tell the user to prompt again
  about 30 minutes later instead of waiting in-turn.
- Runtime-selection v4, 2026-06-21: after Einstein adversarial review and user
  approval for the efficiency follow-up only, the first push attempt was blocked
  locally because the remote push guard rejects dirty payload manifests. The
  full repo quality gate passed (`./scripts/python_quality.sh`, 147 pytest
  tests, 0 type errors/warnings/notes), local commit `753c9db`
  (`Add runtime selection efficiency follow-up`) was created, and Kaggle
  accepted version 4 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection`. The one
  guarded status read returned `KernelWorkerStatus.RUNNING` at
  2026-06-21 00:46:21 -05. On resume, the guarded status read returned
  `KernelWorkerStatus.COMPLETE`, and artifacts were downloaded to
  `runs/kaggle/runtime_selection_v4`. The run failed closed:
  `runtime_proof.status = fail`, no `benchmark/selected_runtime.json`, 18 of 20
  runtime rows passed, and the intended fastest clean row was
  `dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
  at `samples_sec = 25.220604` with zero AMP skips and estimated 10-epoch wall
  time `118950.362625` seconds. The blockers were proof-policy false negatives:
  tiny bounded numerical drift on the selected AMP row and linked proof failures
  from nonselected rows were treated as global blockers.
- Runtime-selection v5, 2026-06-21: after Noether adversarial review of the v4
  result, local commit `fc5227d` (`Relax runtime selection numerical drift
  gate`) scoped linked proof to the selected candidate, accepted only finite
  bounded small numerical drift, kept AMP skips and large drift as row blockers,
  and let skipped AMP rows fail row-local selection without globally rejecting a
  safe alternate row. Focused tests passed (`20 passed`), the full repo gate
  passed (`./scripts/python_quality.sh`, 149 pytest tests and 0 type
  errors/warnings/notes), and local replay of v4 artifacts through the patched
  writer produced proof `pass` for the intended AMP row. Kaggle accepted
  version 5 at
  `https://www.kaggle.com/code/maximusshtefan/eqvae-runtime-selection`; the one
  guarded status read returned `KernelWorkerStatus.RUNNING` at
  2026-06-21 06:14:37 -05. The next approved status read returned
  `KernelWorkerStatus.COMPLETE`, and outputs were downloaded to
  `runs/kaggle/runtime_selection_v5`. Version 5 wrote selected runtime
  `dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
  with `samples_sec = 27.381321`, estimated 10-epoch wall time
  `109563.740875` seconds, zero AMP skips, bounded selected-row numerical drift
  with expected `dual_t4_numerical_delta_failed`, and strict local replay pass.
  It remains blocked from full training launch until real selected-runtime
  debug, checkpoint/resume, and tiny-overfit proofs pass.
- Runtime-selection v6, 2026-06-21: after the compact relaxed-AMP follow-up was
  approved, local commit `580a844` (`Add compact relaxed AMP runtime selection
  follow-up`) passed `./scripts/kaggle_kernel.sh preflight-runtime-selection`
  and Kaggle accepted version 6. Outputs were downloaded to
  `runs/kaggle/runtime_selection_v6`; no `benchmark/selected_runtime.json` was
  written, `runtime_proof.status = fail`, and local replay regenerated the same
  fail-closed proof. The relaxed row
  `dual_t4_ddp__bs12__amp_scalar_gate_relaxed__compile_none__indexed_masked__policy_amp_fp16_scalar_gate_relaxed`
  passed runtime/gate-health with zero AMP skips but reached only `25.288828`
  samples/sec against v5's `27.381321`, so v5 remains the selected-runtime
  fallback.

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

Before any runtime-selection remote push, run the local semantic preflight:

```bash
./scripts/kaggle_kernel.sh preflight-runtime-selection
```

Before any selected-runtime debug/tiny remote push or approval request, run the
local semantic preflight:

```bash
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
```

Before any first full selected-runtime remote push or approval request, run the
local semantic preflight:

```bash
./scripts/kaggle_kernel.sh preflight-selected-runtime-full
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

Push a source-attached real-data kernel only after explicit user permission,
after intentionally accepting the real dataset attachment/setup cost, and after
the relevant push guard is unlocked:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest
```

Future-only selected-runtime debug/tiny push command, not valid while the guard
sees false readiness flags, missing `remote_generate` selector readiness,
missing exact real-dataset metadata, or missing downloaded remote debug/tiny
proof artifacts:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_debug
```

Future-only first full selected-runtime push command, not valid without fresh
explicit user approval and passing `preflight-selected-runtime-full`:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/selected_runtime_full
```

Check remote status after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-selected-runtime-debug
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-selected-runtime-full
```

Download outputs into ignored local run artifacts after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/<real_data_runtime_pretest_run>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-selected-runtime-debug runs/kaggle/<selected_runtime_debug_run>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-selected-runtime-full runs/kaggle/<selected_runtime_full_run>
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
