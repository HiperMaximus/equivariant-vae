# Kaggle CLI Workflow

Status: draft workflow scaffold; synthetic setup-smoke path ready after
permission; synthetic binary timing pretest evidence complete for screening;
capped real-data runtime pretest has a local non-promotable runner/kernel/guard,
upload-simulation proof, and identity/hash/CRC/window plus clean-validation
loader proof plumbing, but remote execution of that lane and later linked
eligibility evidence remain pending; Kaggle source attachments require a
separate confirmation guard
Last updated: 2026-06-19

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
8,192/2,048 spread windows. Timed rows remain ineligible until the linked
compile/DDP/real dataloader-throughput/numerical/corruption/gate-health
evidence is implemented and passes. Remote pushing requires explicit user
permission plus:

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

Push a script kernel only after explicit user permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
```

Check remote status after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
```

Download outputs into ignored local run artifacts after explicit permission:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
```

Pulling from Kaggle can overwrite local files and requires explicit permission:

```bash
KAGGLE_PULL_CONFIRMED=1 ./scripts/kaggle_kernel.sh pull
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
