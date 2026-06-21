# Spec 0003: Kaggle CLI Execution Workflow

Status: draft active workflow scaffold
Implementation readiness: synthetic setup-smoke remote v1 passed as
non-promotable setup evidence; real-data capped smoke has local embedded
packaging/upload-simulation proof and Kaggle source attachments now require
`KAGGLE_FULL_DATASET_CONFIRMED=1`; no-dataset synthetic binary timing pretest
has a committed local kernel/guard implementation and remote v1/v2/v3/v4
completed with non-promotable `synthetic_timing_pass`; v4 is the current
2 GiB-scale repeated-shortlist evidence with per-rank DDP proof; capped
real-data runtime pretest has a local non-promotable runner/kernel/guard and
upload-simulation proof plus identity/hash/CRC/window and clean-validation
loader proof plumbing plus local linked-evidence mechanics/contract scaffolds;
remote v4 passed the canonical real-data proof lane, remote v5 produced two
eligible eager bs4 candidate rows, and remote v6 completed/downloaded with
phase timings plus failed-candidate hash diagnostics but still only two eligible
bs4 rows; remote v7 completed/downloaded, exposed the repeated
failed-candidate exception as `quantile() input tensor is too large`, and
remains non-promotable with no selected runtime. Remote v8 completed/downloaded,
fixed the quantile evidence-plumbing failure, produced six capped-pretest-passing eager
single-visible-T4 bs4/bs8/bs12 FP32 rows, and remains non-promotable with no
selected runtime. The selected-runtime writer plus Kaggle executor/kernel for
`v8_shortlist_eager_amp_then_dual_gate` are locally implemented and guarded as
`runtime_selection_kernel_ready`; runtime-selection v1 was downloaded to
`runs/kaggle/runtime_selection_v1`, proved real dual-T4 DDP timing, refused
selected-runtime writing on linked-proof false negatives, and exposed the local
v2 proof-plumbing/allow-list fix. Runtime-selection v2 was downloaded to
`runs/kaggle/runtime_selection_v2`, fixed those v1 blockers, proved real
dual-T4 DDP timing, and refused selected-runtime writing because
single-visible indexed-mask pass rows lacked candidate-bound gate-health rows.
Runtime-selection v3 was downloaded to `runs/kaggle/runtime_selection_v3` and
wrote `benchmark/selected_runtime.json` selecting
`dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`; full
training/full-run launchers are not Kaggle-push-ready
Owner/workstream: Kaggle GPU execution and artifact retrieval
Last updated: 2026-06-20

## Purpose

Make Kaggle a controlled remote execution surface for GPU runs while keeping this
repo as the source of truth.

Kaggle must not become a second source of canonical model code. Local repo code,
specs, configs, and launchers define the experiment; Kaggle receives generated or
scaffolded script kernels through the Kaggle API.

## Non-Goals

- Do not use Kaggle as a Git remote.
- Do not require Kaggle's GitHub-linked notebook UI workflow.
- Do not edit the historical FSQ notebooks as the new baseline source.
- Do not push a full training or runtime-benchmark kernel before the spec 0001
  launcher is implemented and locally verified. The only current exception is
  the synthetic `kaggle_setup_smoke_ready` setup script, which attaches no real
  dataset and writes non-promotable setup evidence. The capped
  `kaggle_smoke_ready` real-data debug script remains non-promotable, but must
  not be treated as accepted smoke evidence; its source delivery has since been
  migrated locally, and a fresh remote rerun is allowed only when intentionally
  accepting the real dataset attachment/setup cost.
- Do not treat no-dataset synthetic timing output as selected runtime evidence.
  It may screen and order candidates for the later real-data benchmark, but it
  must not write `benchmark/selected_runtime.json`.
- Do not treat the first capped real-data runtime pretest as selected runtime
  evidence. Its local runner/kernel/guard may prove packaging, local
  wrong-accelerator behavior, non-promotable artifact shape,
  identity/hash/CRC/window contracts, clean validation loader plumbing, and
  local linked-evidence mechanics/contract scaffolds. Those scaffolds are not
  row eligibility evidence: compile/DDP remain pending until measured, and
  numerical/corruption CSV rows remain pending unless the exact candidate path
  is covered. Its rows remain ineligible until the linked compile/DDP, real
  dataloader throughput, numerical, corruption, gate-health, graph-break, and
  recompile evidence passes canonically. It must write non-promotable pretest
  artifacts and recommendations, blocked claims, and no
  `benchmark/selected_runtime.json`.
- Do not commit Kaggle credentials, API tokens, output datasets, checkpoints, or
  run artifacts.

## Workflow Contract

The supported workflow is:

```text
repo source -> local Kaggle script kernel folder -> kaggle kernels push
            -> kaggle kernels status/output -> local ignored run artifacts
```

Local commands must go through:

```bash
./scripts/kaggle_kernel.sh
```

Spec 0001 implementation must build the repo code/config payload needed by
Kaggle before remote pushes. The real-data debug scaffold still has a legacy
local sibling-payload build:

```bash
./scripts/kaggle_kernel.sh build
```

The generated `kaggle/kernels/*/payload/` directory is ignored and must not be
committed. It is rebuilt from `src/eqvae`, `configs/spec0001`, `pyproject.toml`,
and `uv.lock`. The 2026-06-13 remote failure proved this sibling payload is not
available to Kaggle script execution through the current CLI path, so it is not
sufficient for rerunning real-data smoke.

The synthetic setup smoke uses a generated single-file launcher instead:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke
```

That command uses `scripts/build_kaggle_embedded_kernel.py` to embed a zipped
payload into ignored `kaggle/kernels/setup_smoke/run.py`. The push guard decodes
that file and verifies the embedded manifest against current source. Kaggle
kernels must not resolve/install dependencies from project metadata unless a
later spec explicitly introduces an offline wheel/bootstrap path.

Remote writes require explicit user permission plus:

```bash
KAGGLE_PUSH_CONFIRMED=1
```

Remote reads/downloads require explicit user permission plus:

```bash
KAGGLE_REMOTE_CONFIRMED=1
```

Remote pulls that can overwrite local kernel files are both remote reads and
local overwrite risks. They require explicit user permission plus both:

```bash
KAGGLE_REMOTE_CONFIRMED=1
KAGGLE_PULL_CONFIRMED=1
```

The current scaffold kernel is:

```text
kaggle/kernels/non_eq_vae_debug
```

It now contains the narrow capped smoke launcher only. It is push-ready only for
local validation of the `kaggle_smoke_ready` debug smoke; the first remote push
failed at import because the sibling payload was not uploaded. Do not rerun it
as accepted remote evidence until rerun with the embedded single-file launcher
and upload-simulation proof. It is not a full benchmark or full-run launcher.

The setup-smoke kernel is:

```text
kaggle/kernels/setup_smoke
```

It is push-ready only for the `kaggle_setup_smoke_ready` setup check after
explicit user permission and `KAGGLE_PUSH_CONFIRMED=1`. It requests no GPU,
attaches no dataset, generates tiny synthetic UBC-format shards under the output
directory, and writes `benchmark/kaggle_setup_smoke.json` as non-promotable
packaging/API/import/artifact evidence.

Remote setup-smoke v1 was pushed on 2026-06-17 and completed with
`status = "smoke_pass"` in the downloaded non-promotable setup artifact. This
proves the current Kaggle API push/status/output path, single-file embedded
payload import, synthetic shard generation, artifact writing, and output
download path. It does not prove real dataset attachment, T4 runtime, loader
throughput, runtime selection, or convergence.

The synthetic timing kernel is:

```text
kaggle/kernels/synthetic_timing
```

It is implemented locally as a generated single-file script kernel. It requests
T4 GPU runtime, attaches no Kaggle sources, generates deterministic UBC-format
binary+CSV shards under the Kaggle working output, and writes only
non-promotable synthetic timing artifacts. It exists to screen and order
candidate rows for the real-data runtime benchmark, including both
`single_visible_t4` and `dual_t4_ddp`, while keeping final runtime selection
blocked until real train/validation shards are measured. Remote v1 completed on
2026-06-18 with the historical compact 0.81 GB profile; downloaded ignored v1
evidence lives under `runs/kaggle/synthetic_timing`. Remote v2 completed on
2026-06-18 with the current 2 GiB-scale profile, but was superseded by remote
v3 because v3 records per-rank DDP device assignment and corrected
`drop_last = false` projection fields. Remote v4 completed the v3 top-four
shortlist with `warmup_steps = 5` and `measured_steps = 25`; all four rows
passed, the repeat policy is marked completed, and no runtime is selected.
Downloaded ignored v4 benchmark evidence lives under
`runs/kaggle/synthetic_timing_repeat_2gib_v4/benchmark`.

The capped real-data runtime pretest scaffold lives in:

```bash
./scripts/kaggle_kernel.sh build kaggle/kernels/real_data_runtime_pretest
```

It is separate from `non_eq_vae_debug` because capped smoke is not a matrix
benchmark, and separate from `synthetic_timing` because it attaches the real
pre-shuffled UBC patch dataset. The kernel uses
`KAGGLE_REAL_DATA_RUNTIME_PRETEST_READY = True`, requests T4 metadata, attaches
only `maximusshtefan/patches-pre-shuffled-ubc-ocean`, requires
`KAGGLE_FULL_DATASET_CONFIRMED=1` before push, and rejects any path that writes
`benchmark/selected_runtime.json`. The local implementation is still
non-promotable: it can prove real-data/local file identity, hashes, CRCs, locked
train/validation windows, split WSI/holdout overlap contracts, and clean
validation loader/collate/normalization plumbing. Tiny fixture roots can only
produce `local_pass`; canonical real `pass` requires the exact dataset slug,
300000/30000 rows, 322/39 WSIs, zero train/validation and masked-holdout
overlap, and the locked 8,192/2,048 spread windows. The local linked-evidence
path can record fixed-window dataloader mechanics, measured `model_forward`
compile/Dynamo counters, real DDP launch proof, candidate batch numerical and
corruption evidence, gate-health rows, phase timings, and failed-candidate
diagnostics, but the capped pretest still cannot select a runtime. Remote v5
proved only two eager bs4 rows; the v6 follow-up code prioritizes eager
single-T4 FP32 train-step evidence by smaller batch size before compiled
diagnostic rows and mirrors candidate/failed evidence counters into
`runtime_proof.json`. Commit `47437a0` was rebuilt, validated, and pushed as
Kaggle version 6 after explicit approval; v6 completed and artifacts were
downloaded to `runs/kaggle/real_data_runtime_pretest_v6`. The launcher and
config allow-lists include `benchmark/phase_timings.json`, and
generated-launcher local simulation validated the full real-data pretest
artifact set before the push. The v6 inspection found no new runtime selection:
only two eager single-T4 bs4 FP32 rows are eligible, bs8/bs12 candidate evidence
failed with hash-only `candidate_train_step_RuntimeError` diagnostics, bs32
eager rows remain OOM, compiled rows remain diagnostic/ineligible, and no
`benchmark/selected_runtime.json` was written.

Remote v2 completed without exercising the real-data proof lane because
`data_root = "auto"` could not resolve Kaggle input files. The local v3 fix
kept auto resolution slug-scoped: only complete shard roots under the expected
`maximusshtefan/patches-pre-shuffled-ubc-ocean` Kaggle mount family can be
selected, while unrelated complete shard roots under `/kaggle/input` are
reported as `complete_unaccepted_candidates`. Remote v3 confirmed the expected
Kaggle shard root and the probe diagnostics, then exposed a separate embedded
payload dependency: the script payload must include
`docs/data/ubc_ocean_masked_holdout_ids.csv` so split/holdout-overlap proof can
run on Kaggle. The pretest writes full
`real_data_proof.data_root_diagnostics` and emits short stderr JSON probe lines
for candidate counts and roots. Rebuild the embedded script after committing
data-root or payload fixes before any remote push.

Remote v4 embedded the masked-holdout CSV and passed the first canonical
real-data proof lane: identity, row counts, WSI/holdout overlap, CRC, locked
windows, and clean validation loader. Its artifacts live under
`runs/kaggle/real_data_runtime_pretest_v4`. The pretest still must not select a
runtime: timed rows remain ineligible until candidate-specific compile/DDP,
real dataloader-throughput, numerical, corruption, and gate-health evidence
passes.

Remote v5 completed on 2026-06-19 and downloaded ignored artifacts under
`runs/kaggle/real_data_runtime_pretest_v5`. It passed canonical real-data,
DDP-launch, dataloader, and gate-health lanes, produced two eligible eager
single-T4 bs4 FP32 rows, left eager bs8/bs12 rows blocked by missing
numerical/corruption evidence coverage, hit runtime OOM for eager bs32 rows, and
wrote no `benchmark/selected_runtime.json`. Compiled `model_forward` rows remain
diagnostic-only/ineligible until full compile-settle coverage exists.

Remote v6 was pushed on 2026-06-19 from commit `47437a0` after explicit user
approval with `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`. Kaggle
accepted version 6 at
`https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`;
an approved status read after the required wait reported
`KernelWorkerStatus.COMPLETE`, and artifacts were downloaded to
`runs/kaggle/real_data_runtime_pretest_v6`. Inspection found
`benchmark/phase_timings.json`, two eligible eager single-T4 bs4 FP32 rows,
five failed candidate evidence attempts with
`candidate_train_step_RuntimeError`, no `benchmark/selected_runtime.json`, and
compiled rows still diagnostic/ineligible. The follow-up v7 diagnostic run
added bounded `failure_message_excerpt` fields to failed candidate evidence so
the actual bs8/bs12 exception could be diagnosed.

Remote v7 was pushed on 2026-06-20 from clean local commit `fea4140` after
explicit user approval and the required
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1` guards. Kaggle
accepted version 7 at
`https://www.kaggle.com/code/maximusshtefan/eqvae-real-data-runtime-pretest`;
approved status reads reported `KernelWorkerStatus.RUNNING` and then
`KernelWorkerStatus.COMPLETE`, and artifacts were downloaded to
`runs/kaggle/real_data_runtime_pretest_v7`. Inspection confirmed the capped
pretest remains non-promotable with no `benchmark/selected_runtime.json`, two
eligible eager single-T4 bs4 FP32 rows, and five failed candidate evidence
attempts now diagnosed as `quantile() input tensor is too large`. The reviewed
local v8 gate-health quantile/evidence-coverage fix is implemented and verified:
it was committed as `614cd95`, rebuilt and validated from a clean source state,
pushed as remote version 8 after explicit approval, completed, and downloaded to
`runs/kaggle/real_data_runtime_pretest_v8`. Inspection confirmed the capped
pretest remains non-promotable with no `benchmark/selected_runtime.json`, zero
failed candidate evidence entries, six capped-pretest-passing eager single-visible-T4 FP32
rows (bs4/bs8/bs12 crossed with `branchless_all` and `indexed_masked`), pending
dual-T4 train-step measurement, and compiled rows still diagnostic/ineligible.
Future remote reads/writes remain permission-gated. The v8 guarded sequence was:
clean committed source state, rebuild, validate, approved
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check`, quota UI
confirmation if the CLI quota endpoint warns, approved
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/real_data_runtime_pretest`,
approved
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest`,
and approved
`KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v8`.
Use the same guards for any rerun or successor remote slice; do not run them
without explicit permission.

The successor selected-runtime benchmark/debug slice must be a separate Kaggle
runtime-selection benchmark, not a v8 promotion. It may use v8 only as
`candidate_shortlist_only` provenance, must record v8 artifact hashes, must
revalidate or write its own linked runtime/dataloader/numerical/corruption/
gate/model-count evidence, and must not write `benchmark/selected_runtime.json`
until real dual-T4 train-step timing and all selected-row safety proofs pass.
The dual-T4 timing gate is required, not optional: it must prove two visible
T4s, `world_size = 2`, `nproc_per_node = 2`, per-rank device binding,
child-process launch command proof, emitted dual-T4 train-step rows, train and
validation dataloader rank coverage, scoped numerical/corruption rows,
gate-health evidence bound to candidate row ids, a hash-linked
`benchmark/stain_corruptor_qa.json`, and global throughput projection. The
selected-runtime writer also enforces 25 measured dataloader batches with
wait/throughput thresholds, three fixed numerical batch indices, train plus
validation clean-RNG corruption rows, per-candidate gate-health rows,
candidate-scoped stain QA, and exact embedded v8 payload membership. Missing,
failed, or skipped dual timing keeps selection blocked.
Local proof plumbing for this slice now exists in
`src/eqvae/benchmarking/runtime_selection.py` with CLI
`src/eqvae/cli/runtime_selection_benchmark.py`. The local default path records
v8 hashes and writes this benchmark's own failed proof/artifact graph, but it
still refuses `benchmark/selected_runtime.json` because no real dual-T4 timing
evidence has been supplied locally. The selected-runtime executor and guarded
Kaggle kernel now live in
`src/eqvae/benchmarking/runtime_selection_executor.py`,
`src/eqvae/cli/runtime_selection_executor.py`, and
`kaggle/kernels/runtime_selection`. The generated runtime-selection kernel
embeds only the required v8 provenance files, validates the
`v8_shortlist_eager_amp_then_dual_gate` slice, runs real selected-runtime
evidence collection on Kaggle, and remains fail-closed if dual timing or linked
proof is missing, failed, or skipped. `runtime_selection_kernel_ready` means the
local build/validate/push guard path is ready; remote status/output/push
commands still require explicit permission and the normal guard variables.
Runtime-selection v1 reached `KernelWorkerStatus.ERROR` after writing benchmark
artifacts. Inspection of `runs/kaggle/runtime_selection_v1` confirmed the
dual-T4 DDP timing gate passed and emitted the required bs4/bs8/bs12 FP32 eager
dual rows, but the strict writer refused `benchmark/selected_runtime.json`
because linked single-visible proof rows were false-negative blocked. The local
v2 fix accepts `benchmark/model_inventory.csv` in the wrapper allow-list,
normalizes `local_pass` gate-health rows before eligibility is computed while
leaving failed non-gate rows ineligible, and requires the
`clean_validation_rng_advanced = false` flag only on validation corruption
rows.
Runtime-selection v2 completed and downloaded to
`runs/kaggle/runtime_selection_v2`; it fixed the wrapper/proof false negatives
and preserved the passing dual-T4 DDP proof, but still refused
`benchmark/selected_runtime.json` because gate-health rows were missing for the
single-visible `indexed_masked` pass rows. The local v3 fix binds branchless
single-visible gate-health rows to same-shape indexed candidates after the
indexed runtime rows have already passed linked evidence.
Runtime-selection v3 completed and downloaded to
`runs/kaggle/runtime_selection_v3`; its proof status is `pass`, dual-T4 DDP
timing proof is `pass`, selected-runtime writing is allowed, and the selected
row is `dual_t4_ddp__bs12__amp_off_fp32__compile_none__indexed_masked`.

For future long-running Kaggle jobs, agents must not wait in-turn after a push
or status read shows a kernel is still `RUNNING` and likely to take more than
about 5 minutes. Give the user a concrete local time to prompt with `continue`,
then stop until the user resumes.

## Kaggle Authentication Contract

Kaggle credentials are local user secrets. They must never be printed, stored in
repo files, or committed.

Use the official Kaggle API authentication paths only, such as local CLI login or
the standard local token file. Agents must ask before running any command that
uses network access or remote Kaggle writes.

## Metadata Contract

Each script kernel folder must contain:

- `kernel-metadata.json`
- exactly one declared `code_file`

Metadata should declare:

- `id`
- `title`
- `code_file`
- `language`
- `kernel_type`
- `is_private`
- `enable_gpu`
- `enable_internet`
- `machine_shape`
- `dataset_sources`
- `competition_sources`
- `kernel_sources`
- `model_sources`

Dataset slugs must be explicit. Do not infer them from display names in the
Kaggle web UI.

The first confirmed training dataset source is:

```text
maximusshtefan/patches-pre-shuffled-ubc-ocean
```

Other historical sources are recorded in
`docs/behavior_inventory_kaggle.md`.

## Acceptance Criteria

This workflow scaffold is complete when:

1. `docs/kaggle_cli_workflow.md` documents the local workflow;
2. `scripts/kaggle_kernel.sh` validates local metadata and guards remote writes;
3. `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json` exists as a private
   script-kernel scaffold;
4. `kaggle/kernels/non_eq_vae_debug/run.py` was initially a non-pushable
   placeholder and has since been replaced by the capped smoke launcher;
5. preflight tracks the Kaggle workflow files;
6. `runs/` is ignored for downloaded Kaggle outputs;
7. `CURRENT.md` records that the scaffold exists but is not push-ready.

This workflow becomes Kaggle-push-ready for the synthetic setup smoke only
after:

1. spec 0001 and the spec index contain `kaggle_setup_smoke_ready`;
2. the generated setup script has `KAGGLE_SETUP_SMOKE_READY = True`;
3. `./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke` has embedded
   the current `src/eqvae`, `configs/spec0001`, `pyproject.toml`, and `uv.lock`;
4. the push guard verifies metadata with empty source lists, no GPU, no internet,
   and a fresh embedded payload manifest;
5. local smoke tests, the upload-simulation test, and the production Python
   quality gate pass;
6. the user explicitly approves the remote write/run.

The real-data capped smoke workflow becomes Kaggle-push-ready only after:

1. spec 0001 and the spec index contain `kaggle_smoke_ready`;
2. the smoke script kernel has `KAGGLE_SMOKE_READY = True`;
3. the launcher source-delivery mechanism is embedded single-file packaging or
   another mechanism proven by upload simulation, not an unuploaded sibling
   payload directory;
4. the payload has a fresh manifest whose git commit and file hashes match the
   current source, and the push guard validates the target kernel ID plus capped
   smoke settings;
5. local smoke tests and the production Python quality gate pass;
6. `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes the
   read-only checks or records only the known quota/files warnings;
7. the user explicitly approves the remote write/run and the 60 GB+ dataset
   attachment/setup cost;
8. the push command includes `KAGGLE_FULL_DATASET_CONFIRMED=1` in addition to
   `KAGGLE_PUSH_CONFIRMED=1`.

The 2026-06-13 real-data smoke push first spent substantial time in Kaggle setup
while attaching the 60 GB+ `patches-pre-shuffled-ubc-ocean` dataset, then ended
with `ModuleNotFoundError: No module named 'eqvae'` because the sibling payload
was not available remotely. The synthetic setup-smoke path now covers setup-only
remote tests: no `dataset_sources`, no real dataset attachment, tiny synthetic
UBC-format shards generated under the output directory, and a separate
non-promotable status/source. Keep the real-data source-attachment guard for
real-data smoke and benchmark kernels. The 2026-06-17 remote-control/timing
planning pass also established that synthetic/random training-time benchmarks
must not attach this real dataset or any other Kaggle source; any kernel
metadata with nonempty `dataset_sources`, `competition_sources`,
`kernel_sources`, or `model_sources` now requires the separate
`KAGGLE_FULL_DATASET_CONFIRMED=1` guard before upload. Real-data spec 0001
debug kernels must use only the known patch dataset source unless a later spec
explicitly authorizes additional sources.

The synthetic binary timing pretest workflow becomes Kaggle-push-ready only
after:

1. spec 0001 and the spec index contain
   `kaggle_synthetic_timing_contract_ready`;
2. the generated script has `KAGGLE_SYNTHETIC_TIMING_READY = True`;
3. `kernel-metadata.json` declares `enable_gpu = "true"`,
   `machine_shape = "NvidiaTeslaT4"`, `enable_internet = "false"`, and empty
   `dataset_sources`, `competition_sources`, `kernel_sources`, and
   `model_sources`;
4. the generated single-file launcher embeds a fresh payload manifest and
   verifies `eqvae` imports from the extracted payload;
5. runtime code clears `EQVAE_DATA_ROOT`, refuses `data_root = "auto"`, writes
   generated shards only under `/kaggle/working`, and asserts resolved
   train/validation paths are under `/kaggle/working`;
6. the streaming synthetic shard writer records profile, seed, byte counts,
   CRCs, file hashes, generation time, free disk, cache state, a pre-timing
   `validate_crc = true` pass for both splits, parsed headers, row counts,
   file sizes, semantic-key uniqueness, sample-id/hash proof from
   `PatchTrainingDataset`, and collate/normalization proof before any measured
   timing row;
7. the pretest writes only
   `benchmark/synthetic_timing_manifest.json`,
   `benchmark/synthetic_timing_runtime_proof.json`,
   `benchmark/synthetic_timing_matrix.csv`, and
   `benchmark/synthetic_timing_recommendations.json` with
   `full_run_eligible = false` and
   `status_scope = "non_promotable_synthetic_timing"` plus a required
   `blocked_claims` object covering final batch size, precision, corruption
   strategy, dataloader settings, single-vs-dual T4 selection, convergence,
   paper evidence, and full-run readiness;
8. the push guard rejects any synthetic timing kernel that attaches Kaggle
   sources or can write `benchmark/selected_runtime.json`;
9. local upload-simulation/import tests and the production Python quality gate
   pass;
10. the user explicitly approves the remote write/run with
    `KAGGLE_PUSH_CONFIRMED=1`. `KAGGLE_FULL_DATASET_CONFIRMED=1` must not be
    required or used for this no-dataset kernel.

The full benchmark/full-run workflow becomes Kaggle-push-ready only after:

1. spec 0001 is locked as implementation-ready;
2. the spec 0001 code/config payload is built into the kernel folder;
3. a full benchmark/full-run launcher replaces the capped smoke launcher;
4. local spec 0001 verification passes;
5. for benchmark kernels, metadata validation requires
   `machine_shape == "NvidiaTeslaT4"` and the safe `single_visible_t4` versus
   `dual_t4_ddp` launch mode recorded in `docs/kaggle_cli_workflow.md`;
6. `KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh api-check` passes the
   required read-only auth/list/status/logs/dataset checks; if the quota
   endpoint warns, GPU quota is verified in the Kaggle web UI and recorded in
   the run notes;
7. the user confirms Kaggle authentication and remote push permission;
8. if metadata includes any nonempty Kaggle source list, the user explicitly
   accepts source attachment/setup cost and the command includes
   `KAGGLE_FULL_DATASET_CONFIRMED=1`.

## Verification Commands

Current scaffold no-network checks:

```bash
./scripts/kaggle_kernel.sh validate
bash -n scripts/kaggle_kernel.sh
python3 -m json.tool kaggle/kernels/non_eq_vae_debug/kernel-metadata.json
./scripts/agent_preflight.sh
```

Spec 0001 post-implementation payload check:

```bash
./scripts/kaggle_kernel.sh build
./scripts/kaggle_kernel.sh build kaggle/kernels/setup_smoke
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" .venv/bin/pytest \
  tests/test_kaggle_smoke.py tests/test_kaggle_embedded_kernel.py
```

Remote commands, only after explicit permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/setup_smoke
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-setup
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-setup
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/real_data_runtime_pretest_v6
```

Synthetic timing remote command, only after adversarial/local verification is
complete and the user explicitly approves the remote write:

```bash
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/synthetic_timing
```

## Known Risks

- A Kaggle GitHub-linked notebook can drift from repo code.
- A script kernel can be pushed without the right dataset slugs if metadata is
  guessed from UI display names.
- Pulling from Kaggle can overwrite local kernel files.
- Enabling internet in Kaggle can hide undeclared dependency and code-source
  assumptions.

## Related Files

- `docs/kaggle_cli_workflow.md`
- `scripts/kaggle_kernel.sh`
- `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json`
- `kaggle/kernels/non_eq_vae_debug/run.py`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0002-strict-python-quality-gate.md`
- `docs/equivariant_vae_transition_plan.md`
