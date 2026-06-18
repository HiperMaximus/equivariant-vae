# Spec 0003: Kaggle CLI Execution Workflow

Status: draft active workflow scaffold
Implementation readiness: synthetic setup-smoke remote v1 passed as
non-promotable setup evidence; real-data capped smoke has local embedded
packaging/upload-simulation proof and Kaggle source attachments now require
`KAGGLE_FULL_DATASET_CONFIRMED=1`; no-dataset synthetic binary timing pretest
has a committed local kernel/guard implementation and remote v1 completed with
non-promotable `synthetic_timing_pass`; full
benchmark/full-run launchers are not Kaggle-push-ready
Owner/workstream: Kaggle GPU execution and artifact retrieval
Last updated: 2026-06-18

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

Remote pulls that can overwrite local kernel files require explicit user
permission plus:

```bash
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
2026-06-18 and downloaded ignored evidence lives under
`runs/kaggle/synthetic_timing`.

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
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output-setup
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
