# Spec 0021: Dual-Model WSI Latent Inference And Smoke Check

Status: locked / implementation-ready
Owner/workstream: held-out embedding generation
Last updated: 2026-08-25

## Purpose

Stream the five Spec 0019 coordinate work shards directly from the official
UBC-OCEAN PNG WSIs through both frozen VAE encoders and write the two Spec 0020
posterior-`mu` stores. Prove that every stored tensor came from the exact
`wsi_id,y,x` patch named by its manifest row, make progress recoverable at WSI
boundaries. A tiny train-only smoke check proves the remote dual-GPU path before
dataset generation. Production has no separately measured limit bundle: the
builder derives each two-model binary size from the frozen work-row count and
rejects any run whose binaries plus a 10 MB metadata reserve exceed Kaggle's
20,000,000,000-byte saved-output cap.

## Non-Goals

- No raw RGB shard, reconstruction, sampled `z`, `logvar`, decoder output,
  classifier, pooling, metric selection, or paper claim.
- No architecture/runtime retuning or pilot candidate search. Inference uses
  the fixed batch-8 synchronous FP32 eager recipe.
- No test-row inference in pilot mode. Production may encode the already-frozen
  test rows because dataset creation is label-free and cannot select rows or
  claims from model outputs; consumers must keep those embeddings sealed.
  Every remote launch still requires fresh explicit authorization.
- No generic WSI or tensor-export framework.

## Immutable Inputs

- The five exact Spec 0019 work manifests and SHA-256 values pinned by Spec
  0020.
- Normal checkpoint
  `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`,
  SHA-256
  `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075`.
- SO(2) checkpoint
  `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`,
  SHA-256
  `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7`.
- The official UBC-OCEAN `train_images/{wsi_id}.png` files.
- One upload-ready local bundle at
  `runs/local/ubc_ocean_latent_input_bundle/`, built by
  `python -m eqvae.cli.build_ubc_latent_input_bundle`. Its exact allow-list is:
  - `dataset-metadata.json`;
  - `spec0021_input_contract.json`;
  - `checkpoints/normal_vae_step_060000.pt` and
    `checkpoints/so2_vae_step_060000.pt`;
  - `manifests/union_patch_manifest.csv`;
  - `manifests/work_shards/run_01_of_05.csv` through `run_05_of_05.csv`;
  - `manifests/task_views/{cancer,tissue}_{train,validation,test}.csv`.
  `spec0021_input_contract.json` has schema
  `spec0021.input_bundle.v1`, status `complete`, the Specs 0019-0021 SHA-256
  values, and every other allow-listed logical path, byte size, and SHA-256.
  It is written last; its own SHA-256 is recorded only by local preflight and
  the later dataset receipt, avoiding self-reference. No timestamp or absolute
  path is allowed.

  Kaggle CLI publication uses a transport envelope, not a second logical
  contract: the upload directory contains only `dataset-metadata.json` and an
  uncompressed `bundle.zip`. The archive contains the 15 contract-bound payload
  files plus `spec0021_input_contract.json` at their exact logical paths. Kaggle
  unpacks this transport archive server-side, so the versioned remote dataset
  must expose the exact original 16-file allow-list and paths. Verification
  pins the ready version, downloads it once, and requires every file's size and
  SHA-256 to equal the local sealed bundle before writing the receipt.

The private dataset ID is
`maximusshtefan/eqvae-ubc-ocean-latent-inputs`. Version 2 is private, ready, and
byte-verified. Its schema-`spec0021.input_dataset_receipt.v1` receipt lives at
`runs/local/ubc_ocean_latent_authority/input_dataset_receipt.json` and binds
contract SHA-256
`6c5a8f011fd0da8157aec136a2ee6963a796f3d1504ecbae0f69d15b65b47136`.
Execution-policy changes do not republish these immutable bytes.

Pilot and production configs are embedded in and hash-bound by the generated
kernel; production depends only on the selected pilot recipe and the frozen
Specs 0019-0021 dataset contracts.
Fresh mode rejects every resume source. Resume mode permits exactly one
additional private dataset named
`maximusshtefan/eqvae-ubc-ocean-latent-run-XX-resume`; every new continuation is a
new immutable version. It contains `spec0021_resume_contract.json` with schema
`spec0021.resume_bundle.v1`, `spec0021_worker_run_XX.resume.json`,
`spec0021_incomplete_run_XX.json`, and for each model exactly one allowed Spec
0020 state, where `{name}` is exactly `normal_vae_mu_run_XX_of_05` or
`so2_vae_mu_run_XX_of_05`:

1. `{name}.bin.partial` plus `{name}.resume.json`;
2. `{name}.bin` plus `{name}.resume.json`, with no completion JSON;
3. `{name}.bin`, `{name}.json`, and stale `{name}.resume.json`;
4. complete `{name}.bin` plus `{name}.json`.

The two model states may differ only when the worker journal proves the accepted
equal-prefix or one-WSI asymmetric windows in this spec. State alone, orphan
binary, partial plus completion JSON, simultaneous partial/final binaries, or
any other file fails. The resume contract pins run, input-dataset receipt,
production config, and every other allow-listed path, size, and hash; its own
hash is pinned by local preflight and the returned dataset receipt. Validate the
read-only attachment in place, copy only its allow-list into one same-directory
working set, fsync, then open the writers.

Every processed PNG records compressed byte size and SHA-256 in per-WSI
evidence. Resume rejects changed bytes. The worker validates all attached file
identities before creating output.

## Frozen Model Contract

1. Hash each checkpoint before `torch.load`.
2. Require checkpoint schema `spec0001.checkpoint.v5`, successful update count
   and optimizer step `60000`, and a `model_state_dict` object.
3. Build through the model registry:
   - `non_eq_vae_translatable`, exactly 3,958,435 learned parameters;
   - `so2_vae_fixed`, exactly 1,180,035 learned parameters and the existing
     fixed-SO(2) identity check.
4. Strictly load the matching state dict; checkpoint/model swaps fail.
5. Set `eval()`, disable gradients, enter `torch.inference_mode()`, and call
   only `encode()`. Persist only finite posterior `mu` converted to CPU FP32
   with shape `(B,16,32,32)`.
6. Input is clean RGB uint8 CHW normalized on-device by the existing exact
   `float32 / 127.5 - 1` function. No corruption, resize, clamp, color transform,
   sampling, or decoder call exists.

## Patch-To-Row Identity Contract

- Validate the exact work-manifest hash, header, increasing atlas identity, and
  strict `wsi_id,y,x` order.
- Open one WSI at a time with pyvips `access="sequential", fail=True`, require
  three bands, group selected coordinates by `y`, and fetch one 256-pixel-high
  strip from the first through last selected `x` for that row.
- Copy each exact HWC RGB crop to contiguous CHW uint8 while keeping its
  `LatentRowIdentity` adjacent through batching and both writer calls.
- Never cross a WSI boundary in a batch.
- Record per WSI a SHA-256 over ordered little-endian
  `struct.pack("<QQQQ", atlas_row_index, wsi_id, x, y)` followed by exactly
  196,608 contiguous CHW uint8 bytes per row.
- The smoke check reopens the PNG independently for indices `0`, `8`, and `15`,
  uses random-access direct crop `(x,y,256,256)` without the grouped strip
  helper, and requires byte equality with the streamed tensors.
- No RGB payload is written to an output artifact.

## Dual-Writer And Resume Contract

- Use one `LatentShardWriter` per model, bound to the same work manifest.
- Ordinary execution submits the same identity batch to both encoders and
  writers.
- At startup, accept only:
  1. identical committed WSI prefixes; or
  2. one model exactly one complete WSI ahead.
- For case 2, reread that WSI and run only the lagging encoder until the writers
  converge, then resume lockstep. Any other prefix relation fails closed.
- Maintain a separate atomic worker-resume JSON. Before either writer receives
  a WSI-final batch, publish `active_wsi` with that WSI's PNG byte
  size/SHA-256 and completed identity-plus-RGB transcript hash, plus the prior
  converged prefix. After both writers commit, atomically promote it into the
  converged evidence list and clear `active_wsi`. The record also binds
  schema/version, input-bundle identity, and run/config/work-manifest hashes.
  Startup accepts only: both writers at the converged prefix with no active WSI;
  or an active journal with both writers at its prior prefix, exactly one writer
  at its end, or both at its end. Reread and match the journaled PNG/transcript,
  run the missing encoder(s), then promote. Every other relation fails closed.
- A recoverable exception or deadline validates and fsyncs both prefixes,
  writes an explicit incomplete marker with each completed WSI list, and exits
  in a publishable state. OOM, worker loss, or forced termination may still
  lose the active WSI.
- A complete run publishes only
  `dataset/{normal_vae,so2_vae}_mu_run_XX_of_05.{bin,json}` and
  `dataset/spec0021_pair_audit_run_XX_of_05.json`. The pair audit has schema
  `spec0021.latent_pair_audit.v1`, binds the run config/input receipt, both
  shard hashes/sizes/sidecars, ordered WSI evidence, and is written last only
  after full CRC32/SHA-256 validation.
- An incomplete run publishes only the exact resume-window files above and
  `dataset/spec0021_incomplete_run_XX.json` with schema
  `spec0021.dual_worker_incomplete.v1`; it never publishes a pair audit.
- Keep complete run payloads remote. Download only an incomplete run's compact
  resume window to `runs/kaggle/ubc_ocean_latents/run_XX_of_05/`. After all five
  pair audits are present remotely, the fixed CPU finalizer validates all ten
  read-only payloads in place, creates six compact location CSVs and twelve
  logical views under `/kaggle/working/dataset/views/`, then atomically writes
  `/kaggle/working/dataset/spec0021_latent_store_global_audit.json` last. That
  JSON has schema `spec0021.latent_store_global_audit.v1` and binds
  the five pair audits, ten shards, six Spec 0019 task manifests, six location
  files, twelve view identities/counts, and shared-row locations. Binary shards
  are never physically merged or copied.

## Pilot Contract

The pilot is a tiny wiring smoke check, not a benchmark, resource projection,
full-WSI rehearsal, or scientific evaluation. It uses only the first 16 frozen
train rows of WSI `15188`; test rows are forbidden. Its fixed recipe is batch
size 8, synchronous D2H, FP32, and eager execution: two batches, one measured
pass, no warmup, no alternative candidates, no compilation, and no autocast.

The partial-WSI smoke iterator reads exactly those 16 rows without hashing or
traversing the complete PNG. Independent random-access crops at smoke indices
`0`, `8`, and `15` must equal the streamed patches. Both frozen checkpoints are
loaded strictly, normal VAE runs on `cuda:0`, SO(2) VAE on `cuda:1`, and exactly
two visible NVIDIA T4 devices are required. Both outputs must be finite CPU
FP32 tensors with shape `(B,16,32,32)` and retain manifest identity/order.

Smoke latents use Spec 0020 FP32 row bytes only in scratch. Exactly 16 rows
produce 1 MiB per model; both files are hash-read and deleted before
publication. They are never canonical shards and receive no completion
sidecars. Production alone performs complete-WSI reads, source-PNG evidence,
resumable shard writes, and final validation.

A successful smoke output contains only
`dataset/spec0021_pilot_matrix.csv` and
`dataset/spec0021_pilot_authority.json`. The matrix has schema
`spec0021.pilot_matrix.v2` and exactly one row for the fixed recipe. The
authority has schema `spec0021.pilot_authority.v2`, status `smoke_passed`,
records WSI `15188`, patch count 16, the input-receipt hash, matrix hash, fixed
recipe and runtime, and is written last. It certifies only
that this narrow path worked; it does not claim production readiness,
throughput, classification quality, or coverage of the full dataset.

The matrix header is exactly:

```text
candidate_index,batch_size,d2h,numeric,execution,status,failure_kind,warmup_passes,measured_passes,patch_count,decode_seconds,h2d_seconds,normalization_seconds,normal_encoder_seconds,so2_encoder_seconds,d2h_seconds,serialization_seconds,fsync_seconds,end_to_end_seconds_p50,patches_per_second_p50,peak_rss_bytes,pinned_bytes,normal_peak_reserved_bytes,so2_peak_reserved_bytes,peak_total_gpu_bytes,output_bytes,finite,identity_match,eligible
```

The single matrix row records decode, transfer, encoder, serialization, fsync,
elapsed time, finite/order checks, RSS, and per-device peak memory. These are
diagnostic observations only and do not select or tune production behavior.
Production uses the same fixed conservative recipe. At complete-WSI boundaries
it keeps a fixed one-hour validation/publication reserve before the configured
session deadline; the smoke result supplies no runtime extrapolation.

Pilot scratch is exactly `/kaggle/working/.spec0021_scratch`; canonical output
is exactly `/kaggle/working/dataset`. The ephemeral FP32 sink and temporary
strips stay under scratch. A `finally` cleanup removes scratch
before output allow-list validation; any remaining scratch path fails the run.
Production uses `/kaggle/working/.spec0021_run_XX_work`; complete runs remove it
after pair-audit publication. Incomplete runs first roll back/fsync the accepted
resume window into `dataset/`, write the incomplete marker last, then delete the
working directory. Nothing outside `dataset/` is publishable.

## Modes And Authorization Gates

- `local-fixture`: fake/array WSI reader, CPU models, synthetic manifests; local
  mechanics only.
- `pilot`: exactly WSI 15188 from the train split; no test coordinate may be
  decoded or encoded.
- `production`: one exact run `01..05`; requires:
  - a matching 16-patch smoke result and the fixed worker config;
  - exact Specs 0019-0021 manifest/storage/inference hashes in the config;
  - immutable input bundle identity;
  - fresh explicit user authorization and Kaggle confirmation variables.

Tracked kernel sources are
`kaggle/kernels/ubc_ocean_latent_inference/run_template.py` and
`kaggle/kernels/ubc_ocean_latent_finalizer/run_template.py`. Generated upload
directories are ignored under `runs/local/ubc_ocean_latent_kernels/`:

| Mode | Directory | Kernel ID |
| --- | --- | --- |
| pilot | `pilot/` | `maximusshtefan/eqvae-ubc-ocean-latent-pilot` |
| run 01..05 | `run_XX/` | `maximusshtefan/eqvae-ubc-ocean-latent-run-XX` |
| finalizer | `finalizer/` | `maximusshtefan/eqvae-ubc-ocean-latent-finalize` |

Each contains only `kernel-metadata.json`, generated `run.py`, and
`spec0021_inference_config.json`. Pilot/run configs use schema
`spec0021.inference_config.v2`. Fresh inference metadata attaches exactly
UBC-OCEAN and the immutable input dataset. Resume reuses the same run-specific
kernel ID in a new kernel version and adds exactly its run-specific resume
dataset. Inference uses one controller, two T4s, Internet only for the locked
Torch/libvips bootstrap, and no kernel/model sources. `run.py` embeds the config,
source payload, Spec 0021 hash, input receipt, and for production the accepted
pilot authority hash. Public code lazy-loads pyvips through the
typed seam. The finalizer is CPU-only with Internet enabled solely to upgrade to
the latest PyPI Torch before importing project code. It attaches the immutable
input dataset plus exactly the five run kernel outputs, and its same-named config
uses `spec0021.finalizer_config.v1` to bind the input receipt and five exact
production configs. The global audit records the resolved Python, Torch, and
Torch CUDA-build versions.

The config's exact top-level fields are `schema_version`, `mode`, `run_number`,
`spec_sha256`, `input_dataset_receipt`, `input_contract_sha256`,
`work_manifest_sha256`, `normal_checkpoint_sha256`, `so2_checkpoint_sha256`,
`pilot_authority_sha256`, `selected_recipe`, `expected_binary_output_bytes`,
`saved_output_limit_bytes`, and `resume_dataset_receipt`. Pilot uses null for
`run_number`, the pilot-authority hash, `selected_recipe`, both output-size
fields, and the resume receipt. Before input publication only, pilot may also have a null
`input_dataset_receipt`; this is a staged-template preflight state and the push
guard must reject it. The post-publication pilot and all production configs
require the receipt. Fresh production uses only a null resume receipt; resume
requires it.
Canonical JSON is UTF-8, sorted keys,
compact separators, and one trailing newline.

Authorization is staged and never inherited:

1. Local staging performs no network access.
2. Explicit input-publication authorization permits only creating/versioning
   `eqvae-ubc-ocean-latent-inputs` and verifying its listing; it does not permit
   a kernel push.
3. After the receipt is pinned and pilot rebuilt locally, fresh pilot
   authorization permits only pilot API-check/push/status/output.
4. After local pilot validation and production rebuild, one fresh authorization
   may name all five exact fresh run IDs; it permits only those five pushes and
   their status/output reads.
5. Every resume dataset publication and run-specific resume push requires a new
   authorization naming that run and dataset version.
6. After all five pair audits exist, finalizer build/preflight remains local.
   A separate authorization naming only
   `maximusshtefan/eqvae-ubc-ocean-latent-finalize` permits its push and output
   read; no production rerun is implied.

No remote operation is authorized by this spec or by completion of a prior
stage.

## Outputs

- Public frozen-checkpoint loader.
- Public manifest/WSI batch stream used by the worker.
- `src/eqvae/cli/generate_ubc_latent_stores.py`: worker/controller CLI.
- `src/eqvae/cli/build_ubc_latent_input_bundle.py`: exact input bundle builder.
- `src/eqvae/cli/build_ubc_latent_kernel.py`: pilot/run/finalizer
  upload-directory builder.
- `src/eqvae/cli/finalize_ubc_latent_stores.py`: ten-shard validator, compact
  split-view writer, and global audit-last publisher.
- `kaggle/kernels/ubc_ocean_latent_inference/run_template.py` plus guarded
  `scripts/kaggle_kernel.sh` build/preflight/publication branches.
- `kaggle/kernels/ubc_ocean_latent_finalizer/run_template.py`: disk-safe CPU
  finalizer over five read-only kernel-output sources.
- One-row smoke matrix, tiny-pilot authority, and fixed production config.
- Five normal and five SO(2) Spec 0020 shards only after authorized production.

## Executable Workflow

The generated wrapper invokes exactly one of:

```bash
python -m eqvae.cli.generate_ubc_latent_stores pilot \
  --config spec0021_inference_config.json \
  --payload-root /kaggle/working/.spec0021_private/payload \
  --output-root /kaggle/working/dataset \
  --scratch-root /kaggle/working/.spec0021_scratch
python -m eqvae.cli.generate_ubc_latent_stores production \
  --config spec0021_inference_config.json \
  --payload-root /kaggle/working/.spec0021_private/payload \
  --output-root /kaggle/working/dataset \
  --scratch-root /kaggle/working/.spec0021_run_XX_work
```

The production command owns full shard validation and pair-audit-last
publication; there is no separate command that can attest an incomplete pair.

The private immutable-input dataset version 2 is already published and
byte-verified. Its Spec 0021 hash inside
`input_bundle.SPECIFICATION_SHA256` is frozen creation provenance for those
sealed bytes; later execution-policy edits do not mutate or invalidate that
dataset. Every generated kernel separately pins the current Spec 0021 hash, and
the input receipt must match the local input-contract SHA-256. Do not rebuild or
republish the input dataset for a smoke-policy-only change.

Local smoke preparation ends only after these focused commands pass:

```bash
./scripts/kaggle_kernel.sh build-latent-inference pilot
./scripts/kaggle_kernel.sh preflight-latent-inference pilot
.venv/bin/python -m pytest -q \
  tests/test_spec0021_inference_core.py::test_manifest_array_stream_preserves_sparse_rgb_identity_and_transcript \
  tests/test_spec0021_worker_pilot.py::test_tiny_pilot_pass_reads_and_encodes_exactly_two_batches \
  tests/test_spec0021_inference_workflow.py::test_pilot_builder_binds_exact_metadata_config_and_wrapper
```

With separate pilot authorization, use generic guarded push/status/output on
`runs/local/ubc_ocean_latent_kernels/pilot`, downloading to
`runs/kaggle/ubc_ocean_latent_pilot`. Validate the one 16-patch smoke row,
authority binding, and scratch absence locally:

```bash
.venv/bin/python -m eqvae.cli.generate_ubc_latent_stores validate-pilot \
  --input runs/kaggle/ubc_ocean_latent_pilot \
  --output runs/local/ubc_ocean_latent_authority/pilot_receipt.json
./scripts/kaggle_kernel.sh build-latent-inference production-all
./scripts/kaggle_kernel.sh preflight-latent-inference production-all
```

Production needs no limits file or additional probe. The build pins Kaggle's
20,000,000,000-byte saved-output cap and rejects a run unless its two exact
`64 + rows * 65,536`-byte binaries plus a 10 MB reserve fit. Run 01 has 121,199
rows and therefore 15,885,795,456 binary bytes, leaving 4,114,204,544 bytes
before the cap; its JSON sidecars and pair audit fit inside the reserve.

After fresh authorization naming all five runs, push the five generated
directories independently. Complete payloads remain remote for the finalizer.
Only an incomplete run is downloaded, validated, and staged with:

```bash
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output \
  maximusshtefan/eqvae-ubc-ocean-latent-run-XX \
  runs/kaggle/ubc_ocean_latents/run_XX_of_05
./scripts/kaggle_kernel.sh build-latent-resume XX \
  runs/kaggle/ubc_ocean_latents/run_XX_of_05
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_DATASET_WRITE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh publish-latent-resume XX
KAGGLE_REMOTE_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh verify-latent-resume XX
./scripts/kaggle_kernel.sh build-latent-inference resume XX
./scripts/kaggle_kernel.sh preflight-latent-inference run-XX
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh push \
  runs/local/ubc_ocean_latent_kernels/run_XX
```

Run the publication pair only after separate resume-dataset authorization and
the final push only after separate resume-kernel authorization. The rebuild
must bind the returned dataset version. Never combine files from different
attempts.

The ten FP32 payloads total about 73.17 GiB while the local filesystem has only
about 33 GiB free, so they must not be downloaded or copied into one local
pair-root. When all five pair audits are complete remotely, build and preflight
the CPU finalizer. It attaches the immutable input dataset plus exactly the five
production kernel outputs, symlinks their read-only mount directories into a
temporary pair-root, scans all ten payloads in place, and publishes only the six
location CSVs plus global audit:

```bash
./scripts/kaggle_kernel.sh build-latent-inference finalizer
./scripts/kaggle_kernel.sh preflight-latent-inference finalizer
KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 \
  ./scripts/kaggle_kernel.sh push \
  runs/local/ubc_ocean_latent_kernels/finalizer
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh status \
  maximusshtefan/eqvae-ubc-ocean-latent-finalize
KAGGLE_REMOTE_CONFIRMED=1 ./scripts/kaggle_kernel.sh output \
  maximusshtefan/eqvae-ubc-ocean-latent-finalize \
  runs/kaggle/ubc_ocean_latent_store_finalizer
```

The push/output commands require their own explicit finalizer authorization.
Success means the downloaded global audit exists, has status `complete`, binds
ten fully scanned read-only shards and all twelve split views, and accompanies
exactly six location CSVs. A partial set, location CSVs without the audit, or an
audit written before any dependency is not complete.

## Local Implementation Status

Implemented: hash-first final-checkpoint loading, public hash-bound work
manifests, typed/lazy WSI streaming, exact CHW transcript evidence, durable
writer-prefix rollback, and the dual-writer active-WSI journal with verified
one-model catch-up and both-committed recovery. Local fixtures cover coordinate
identity and interruption windows.

Remote pilot kernel version 1 completed and validated locally: exactly 16 train
rows from WSI 15188, two batch-8 encoder calls, finite identity-matched outputs,
and 1 MiB of temporary FP32 latents per model. Its accepted authority SHA-256 is
`a59c1232687c93a6e5638a699275a5f91f78dba1bffb53c8fb462b3cfe41474e` and
its matrix SHA-256 is
`3b043c52306fb13cc506629a615686b8ca8cdaf448f09ee9994de5ecea3968f8`.
Pending remotely: five production runs or authorized resumes and the global
finalizer. The immutable input dataset is already published and byte-verified.
No test embedding has been consumed.

## Acceptance Criteria

1. Real final checkpoints hash, load strictly into only their expected model,
   and produce finite FP32 `(B,16,32,32)` direct-`encode` output.
2. Coordinate-coded fake WSIs prove exact patch/identity pairing, RGB/CHW
   convention, sparse-strip offsets, bounds, and queue order.
3. The same identity batch reaches both encoders and both writers; no model may
   see a different valid patch set.
4. Interruptions inside a WSI, after one writer commits, after binary rename,
   and before pair-audit publication recover to byte-identical stores or fail
   closed as specified.
5. Pilot refuses test rows, reads only the first 16 rows of train WSI 15188,
   matches direct-crop sentinels `0`, `8`, and `15`, emits one fixed-recipe row,
   proves finite aligned FP32 outputs, and removes its temporary payloads.
6. Production refuses a missing/changed dataset-contract hash, immutable input,
   pilot, work manifest, checkpoint, PNG identity, or remote authorization.
7. Pair completion validates both shards fully and publishes the audit last;
   final global validation proves all ten shards, six task-manifest/location
   pairs, and twelve model/task/split views before publishing its audit last.
8. No raw RGB, decoder output, sampled latent, test metric, or classifier
   artifact is written.
9. Fresh/resume guards enforce the exact staged authorization and source
   allow-lists; every allowed Spec 0020 resume window succeeds and every other
   file/state combination fails before an encoder runs.
10. Focused tests, full Python quality, `git diff --check`, repo/workspace
    preflights, generated-kernel simulations, and clean-context implementation
    review pass.

## Known Risks

- Two model writers have an asymmetric WSI commit window; the one-WSI catch-up
  rule is mandatory.
- Full output is about 36.58 GiB per model. Run 01 is the largest two-model run
  at 15,885,795,456 binary bytes (14.795 GiB); the build must keep its 10 MB
  metadata reserve below Kaggle's 20,000,000,000-byte saved-output cap.
- The smoke check says nothing about full-run duration or worst-case PNG decode.
  A failed run must be treated as incomplete and may resume only through the
  existing WSI-boundary receipt workflow.
- Full payload validation rereads GiB of output and must be budgeted before
  publication.

## Related Files

- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0019-task-consumption-manifests-and-kaggle-work-plan.md`
- `docs/specs/0020-fp32-latent-shard-format-and-io.md`
- `src/eqvae/data/latent_shards.py`
- `kaggle/generate_ubc_ocean_test.py`

## Review Record

Clean-context review on 2026-08-25 removed full-WSI pilot traversal, candidate
tuning, all-PNG audit, and pilot-derived resource projections. It retained the
production complete-WSI evidence/resume contract, separated current execution
spec binding from immutable v2 creation provenance, and replaced unused
production-limit inputs with a derived saved-output-cap check before run 01.
