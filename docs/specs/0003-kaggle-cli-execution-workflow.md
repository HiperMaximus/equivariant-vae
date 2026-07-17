# Spec 0003: Kaggle CLI Execution Workflow

Status: draft active workflow scaffold.

Selected runtime (locked fallback): runtime-selection v5 wrote
`benchmark/selected_runtime.json` selecting
`dual_t4_ddp__bs12__amp_conservative__compile_none__indexed_masked__policy_amp_fp16_conservative`
— dual-T4 bs12 AMP-conservative, zero AMP skips, ~30.4 h projected for 10 epochs.
Spec 0008's capped selected-runtime debug/tiny gate passed strict
downloaded-output verification for v5 (`runs/kaggle/selected_runtime_debug_v5`),
but that gate is intentionally non-promotable and is not a full-run launcher:
Spec 0009 owns the guarded full-run kernel, exact 10-epoch schedule,
validation/checkpoint cadence, resume hardening, verifier, and approval gate.
Spec 0011 re-derives the runtime as a reusable per-(model × hardware) search;
v5 stays the fallback until that generator emits a new `selected_runtime.json`.

The historical FSQ script is runtime reference material only (launch/env, loader,
AMP, DDP, compile, layout, and checkpoint hypotheses); it is not a source for FSQ
quantization, PixelShuffle/sub-pixel upsampling, final `tanh` bounding, the exact
old corruptor, or `rot90`/discrete-latent equivariance artifacts.

Owner/workstream: Kaggle GPU execution and artifact retrieval
Last updated: 2026-07-16

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
blocked until real train/validation shards are measured.

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
diagnostics, but the capped pretest still cannot select a runtime.

Real-data resolution is slug-scoped: `data_root = "auto"` selects only complete
shard roots under the expected `maximusshtefan/patches-pre-shuffled-ubc-ocean`
Kaggle mount family, reports unrelated complete roots under `/kaggle/input` as
`complete_unaccepted_candidates`, writes full
`real_data_proof.data_root_diagnostics`, and emits short stderr JSON probe lines
for candidate counts and roots. The embedded script payload must include
`docs/data/ubc_ocean_masked_holdout_ids.csv` so the split/holdout-overlap proof
can run on Kaggle; rebuild the embedded script after any data-root or payload
fix before a remote push. The pretest still must not select a runtime: timed
rows remain ineligible until candidate-specific compile/DDP, real
dataloader-throughput, numerical, corruption, and gate-health evidence passes.
Use the same permission gates for any rerun or successor remote slice.

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
Approval for the efficiency/runtime-selection follow-up is separate from
approval for the first real full training run (multiple dozen hours long):
agents must not ask to launch that run until implementation, environment
checks, efficiency decisions, selected-runtime debug/resume, artifact checks,
tiny-overfit, and gate-health checks are complete.
Future runtime optimization/debug work should preserve the FSQ-derived runtime
hypotheses without copying the FSQ implementation: launch/env settings
(`torchrun --standalone
--nproc_per_node=2`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), per-rank
CUDA/NCCL binding, mmap/sequential-read and pinned non-blocking H2D behavior,
channels-last, cuDNN nondeterministic benchmark mode, AMP with FP32 loss islands,
stable compile warmup, DDP fast paths, optimizer/zero-grad fast paths, and
checkpoint/resume state. It must also preserve the newer spec corrections: clean
validation must not execute the corruptor or consume corruption RNG, repeated
AMP skips block the row, and schedule/checkpoint cadence is driven by successful
optimizer updates only.
Before any future runtime-selection remote push or remote-write approval
request, run the cheap local semantic preflight:

```bash
./scripts/kaggle_kernel.sh preflight-runtime-selection
```

This preflight builds and validates the generated kernel, runs the
runtime-selection writer tests, exercises the generated wrapper's import and
fail-closed artifact-validation paths, and replays downloaded v5 artifacts when
they are present locally.
The Spec 0006 local selected-runtime mechanics slice is implemented and locally
verified using `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`.
It remains non-promotable and cannot unlock a long real run or remote push by
itself. The next workflow is split into Spec 0007 for the real UBC/DDP/AMP
selected-runtime runner and Spec 0008 for canonical fixed-32 selector generation
plus the narrow selected-runtime debug/tiny Kaggle push-readiness milestone. Do
not ask for the first long real training launch until the downloaded real
debug/resume/artifact/gate-health/tiny-overfit proof artifacts pass.

Selected-runtime debug/tiny gate workflow:

- build target: `kaggle/kernels/selected_runtime_debug`;
- CLI contract target: `python -m eqvae.cli.selected_runtime_gate`;
- embedded selected-runtime input:
  `runs/kaggle/runtime_selection_v5/benchmark/selected_runtime.json`;
- local semantic preflight before any selected-runtime debug/tiny push or
  approval request:

```bash
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
```

This preflight builds and validates the generated selected-runtime debug kernel,
runs the selected-runtime gate tests, runs the generated-wrapper import-only
simulation, and runs the full local fail-closed artifact simulation. Passing
this preflight means the contract is locally coherent and wrapper-buildable; it
does not mean the real gate passed. The push guard remains stricter: it
requires explicit user permission plus
`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1`, exact
metadata/source lists, fresh embedded payload verification, the
`selected_runtime_debug_gate_contract_ready` docs tokens, the structured
`eqvae.cli.selected_runtime_gate --verify-push-ready` check, Spec 0008
`remote_generate` generator/readiness pass, and exact real-dataset metadata. In this
mode the approved remote kernel, not local pre-push state, generates and
validates the canonical fixed-32 selector from the real Kaggle train shard
before training.

Next workflow slice before any selected-runtime debug/tiny remote action:

1. Keep the Spec 0007 runner and Spec 0006 local mechanics as the fail-closed
   local baseline.
2. Keep the selected-runtime debug kernel push guard blocked.
   `remote_pass_ready` and `fixed_32_selector_real` remain false until
   structured real proof artifacts and downloaded remote artifacts prove the
   canonical real fixed-32 selector.
3. Implement Spec 0008 next: local-first fixed-32 selector generation, synthetic
   selector rejection as non-canonical, canonical real selector validation, and
   artifact-derived push readiness.
4. A future agent may ask for explicit approval for the narrow
   selected-runtime debug/tiny remote push only after
   `./scripts/kaggle_kernel.sh preflight-selected-runtime-runner`,
   `./scripts/kaggle_kernel.sh preflight-selected-runtime-debug`, and the
   Spec 0008 `remote_generate` readiness verifier all pass, the shell push
   guard is expected to pass, and exact real-dataset metadata is attached. In
   this mode the approved remote kernel
   generates and validates the canonical fixed-32 selector from the real Kaggle
   train shard before training. That approval is not approval for the long
   training run. Remote commands still require
   `KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1` for writes and
   `KAGGLE_REMOTE_CONFIRMED=1` for status/output reads.
5. The first full training run (multiple dozen hours long) remains prohibited
   until downloaded selected-runtime debug, checkpoint/resume,
   artifact-manifest, gate-health, and tiny-overfit proofs all pass. If they
   pass, the first full real selected-runtime run becomes the immediate next
   candidate action.

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

If local Kaggle CLI OAuth credentials exist, `scripts/kaggle_kernel.sh` routes
authenticated reads/writes through `scripts/kaggle_oauth_exec.py` by default.
That helper generates a fresh short-lived OAuth access token from the installed
Kaggle SDK, passes it to the child CLI through a temporary 0600 token file, and
deletes the file when the child exits. This is specifically to avoid stale
cached `credentials.json` access tokens and shell token-substitution leaks. Use
`KAGGLE_DISABLE_FRESH_OAUTH=1` only when intentionally debugging the raw Kaggle
CLI auth path. `api-check` must prove authentication through wrapped endpoint
calls; it must not depend on a separate raw `kaggle auth print-access-token`
probe.

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
   `machine_shape = "NvidiaTeslaT4"`, `enable_internet = "true"` (for the
   decision-0012 runtime torch upgrade only), and empty `dataset_sources`,
   `competition_sources`, `kernel_sources`, and `model_sources`;
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

Runtime-selection local semantic preflight before any successor push:

```bash
./scripts/kaggle_kernel.sh preflight-runtime-selection
```

Selected-runtime debug/tiny local semantic preflight before any gate push or
approval request:

```bash
./scripts/kaggle_kernel.sh preflight-selected-runtime-debug
```

Remote commands, only after explicit permission:

```bash
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh push
KAGGLE_PUSH_CONFIRMED=1 ./scripts/kaggle_kernel.sh push kaggle/kernels/setup_smoke
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-setup
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-real-data-runtime-pretest
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status-selected-runtime-debug
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-setup
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-real-data-runtime-pretest runs/kaggle/<real_data_runtime_pretest_run>
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-selected-runtime-debug runs/kaggle/<selected_runtime_debug_run>
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
  assumptions. The run kernels enable it only to `pip install --upgrade` the torch
  stack (decision 0012); the embedded payload and empty source lists still bar any
  undeclared CODE source.

The empty source lists and the embedded payload are the durable faces of the
hermeticity invariant, not leftover caution. Internet was originally the third face
(off); the run kernels now enable it solely for the runtime torch upgrade
(`docs/decisions/0012-kaggle-runtime-torch-upgrade.md`), which keeps the
embedded-code and empty-source faces intact and trades only torch-version pinning.
The code-delivery mechanism and the planned future path (pip-install from a pinned
commit once the GitHub repo is public — not `dataset_sources`) are recorded in
`docs/decisions/0011-kaggle-code-delivery.md`.

## Related Files

- `docs/kaggle_cli_workflow.md`
- `scripts/kaggle_kernel.sh`
- `kaggle/kernels/non_eq_vae_debug/kernel-metadata.json`
- `kaggle/kernels/non_eq_vae_debug/run.py`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0002-strict-python-quality-gate.md`
- `docs/equivariant_vae_transition_plan.md`
