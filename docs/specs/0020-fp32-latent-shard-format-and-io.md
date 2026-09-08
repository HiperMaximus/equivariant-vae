# Spec 0020: FP32 Latent Shard Format And I/O

Status: storage and twelve-view integration implemented; real base shards
complete; Spec 0021 final global audit pending
Implementation readiness: accepted after independent adversarial contract review
Owner/workstream: held-out embedding transport
Last updated: 2026-08-24

## Purpose

Adapt the proven fixed-record UBC patch `.bin` mechanics to the deterministic
posterior means produced by both frozen VAEs. The result is a small reusable
storage layer that writes and memory-maps full FP32 `(16,32,32)` latent tensors
without materializing the 109.75 GiB RGB patch union.

Each model has five physical latent shards aligned row-for-row to the five Spec
0019 union work manifests. Task and split membership stay in compact manifests,
so the two model stores expose twelve logical model-by-task-by-split views
without duplicating any FP32 tensor.

## Non-Goals

- No WSI reading, model construction, checkpoint loading, inference, Kaggle
  kernel, dataset upload, or remote launch.
- No classifier, WSI aggregation, pooling, metric, uncertainty, or sealed-test
  access decision. A later evaluation spec must lock those before test use.
- No FP16, sampled `z`, `logvar`, compressed container, variable record shape,
  raw patch output, or physical concatenation of the five shards.
- No general tensor serialization framework. This format exists for one fixed
  posterior-`mu` contract.

## Inputs And Data Contract

- Source manifests:
  `runs/local/ubc_ocean_eval_consumption/work_shards/run_XX_of_05.csv` from
  Spec 0019. The writer and reader require the independently expected SHA-256;
  they never self-attest whatever file happens to be present. Trusted values are:

  | Run | Manifest SHA-256 |
  | ---: | --- |
  | 1 | `76cc5f9b86b75b9e46250c80e7b5c98d2f0451a12b38665ae45b055767b7a456` |
  | 2 | `11d21482c9d3e083bc5138973c6b09f6b659e7d753853c0290382320e78e6200` |
  | 3 | `9414f638abc24963e821de8bbedccaf294a14a7ea768a01687d5b105e634402b` |
  | 4 | `e7c5d8d08996e3bac440b5779547b2bbeb4443e7481dcb74998a87aa3054697f` |
  | 5 | `5485c44ababeaba9a4e2cc75a1a6b2927d89f10ed83a121dfcb81a68cfc23d6a` |
- Source rows have the exact Spec 0019 union header and strict ascending
  `wsi_id,y,x` order. CSV row position is the binary `file_index`.
- Models are exactly `normal_vae` and `so2_vae`. Required trusted checkpoint
  hashes are normal
  `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075`
  and SO(2)
  `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7`.
  The writer and reader compare the configured model/hash pair to these expected
  values before accepting provenance.
- Every record is deterministic posterior `mu`, CPU, finite FP32, shape
  `(16,32,32)`. Non-contiguous or channels-last CPU FP32 input is converted to
  standard contiguous CHW before explicit little-endian serialization; dtype,
  device or shape mismatch fails.
- One record contains 16,384 values and exactly 65,536 payload bytes.

## Binary Contract

The header remains exactly 64 bytes and uses little-endian struct format:

```text
<8sIQiiii4s3s21x
```

| Field | Value |
| --- | --- |
| magic | `EQV_LATN` |
| payload CRC32 | unsigned CRC32 over bytes after the header |
| tensor count | unsigned 64-bit record count |
| channels | `16` |
| height | `32` |
| width | `32` |
| version | `1` |
| dtype | `F32L` |
| layout | `CHW` |

Record `i` begins at `64 + i * 65536`. The exact final file size is therefore
`64 + tensor_count * 65536`. The distinct magic prevents the existing uint8
patch loader from interpreting latent tensors as images.

## Sidecars And Completion Contract

Each final `{model}_mu_run_XX_of_05.bin` has one same-stem JSON completion
sidecar written atomically after the final binary exists. It records:

- schema/status, model name, checkpoint SHA-256;
- source-manifest logical basename, SHA-256, run number, row count, first/last
  exact row identity;
- tensor dtype, shape, layout, record bytes and count;
- payload CRC32, payload SHA-256 and exact file size;
- ordered completed WSI IDs and count.

Absolute paths and timestamps are forbidden so uninterrupted and resumed output
can be byte-identical and portable. The sidecar is the completion marker. A
`.bin` without its valid JSON is not a published shard. Task labels and
membership are not copied into the binary or JSON; the pinned union manifest is
their single source of truth. JSON uses UTF-8, sorted keys, two-space indentation
and one trailing newline.

## Resumable Writer Contract

- Build through same-directory `.partial` binary and `.resume.json` state. The
  provisional header has the final magic/version/dtype/layout/shape with zero
  count and CRC; rewriting it with final count/CRC is idempotent.
- Reserve the 64-byte header, then append batches only for the next manifest
  WSI. One WSI must be completed before another begins. Every append supplies
  the expected zero-based `row_start` plus the exact ordered manifest identities
  `(atlas_row_index,wsi_id,x,y)` for its tensor batch. The writer compares them
  to the next manifest slice, preventing a within-WSI tensor swap from passing
  through counts alone. `file_index` is the zero-based CSV data-row position;
  the CSV header is not a row. This proves declared storage order; the later
  inference worker must prove that each declared identity accompanied the patch
  actually fed to the encoder.
- After exactly the manifest row count for a WSI is written, flush and fsync the
  binary, then atomically publish state with the committed row/byte boundary
  `64 + committed_rows * 65536`,
  ordered completed WSI IDs, prefix CRC32, prefix SHA-256, manifest hash, model,
  checkpoint, pinned union hash and tensor contract. Committed payload bytes are
  always exactly `committed_rows * 65536`; the completed WSI list must be the
  exact manifest prefix whose derived row count equals `committed_rows`.
- An interrupted active WSI is not committed. Resume verifies the state binding,
  rescans payload bytes after the header through the committed boundary to
  reproduce its CRC32/SHA-256, truncates any
  trailing bytes to the committed boundary, and restarts with the next WSI.
- Finalization requires every manifest row exactly once. It writes the final
  header and fsyncs, atomically renames the partial binary, writes the completion
  JSON as the last published artifact, then removes private resume state.
  Finalization/recovery must be idempotent if interruption occurs after the
  final header or binary rename.
- Keep resume state until completion JSON publication. Recovery is exact:
  - partial binary plus an all-rows-committed state finalizes without inference;
  - final binary plus state but no JSON validates and publishes JSON;
  - valid final binary plus JSON and stale state validates then removes state;
  - an orphan final binary without JSON or state fails closed;
  - a partial binary/state mismatch or either one alone fails closed;
  - simultaneous partial and final binaries fail closed.
- Existing complete outputs fail closed unless they validate exactly; the
  writer never silently overwrites them.

## Memory-Mapped Reader Contract

- Validate magic/version/dtype/layout/shape, exact file size, sidecar binding,
  source-manifest hash and row count before exposing records.
- `LatentTensorDataset(..., validate_payload=False)` performs lightweight
  structural/provenance validation without rereading GiB of payload.
  `validate_latent_artifact(...)` performs the full CRC32 and SHA-256 scan and
  is mandatory at publication/intake; it catches same-size payload mutation.
- Return a zero-copy read-only FP32 tensor view shaped `(16,32,32)` using the
  fixed offset. Worker pickling drops process-local file and mmap handles.
- Support `sequential`, `random`, and `none` mmap advice. WSI aggregation uses
  sequential advice; shuffled tissue-index training may use random/default
  advice without rewriting the ordered binary.
- Returned tensors are zero-copy views over an OS-read-only mapping. Mutation is
  unsupported and may fault; callers needing writable storage must clone before
  the dataset is closed. Closing with live views is caller misuse.

## Store And Task-View Validation

`validate_latent_store_pair` accepts the independently pinned union manifest,
five expected work-manifest hashes, and the two expected checkpoint hashes. It
requires exactly runs 1 through 5 for each model, with no duplicate or missing
run; identical shape/dtype/layout; the correct model/checkpoint/manifest binding;
and row-for-row shard concatenation equal to the pinned 599,398-row union in
strict order.

The six authoritative Spec 0019 task manifests and frozen identities are:

| Task | Split | Rows | SHA-256 |
| --- | --- | ---: | --- |
| cancer | train | 314,755 | `03e8e0d7aefb8d30d049c8521c44ebc88ca3709ea26a65765b772df2b3af65fb` |
| cancer | validation | 68,045 | `3d96c739320ab1facbdf2bb93b18fc8d2c951a7f30a3d61f5ce3587e4c402ce5` |
| cancer | test | 67,138 | `1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc` |
| tissue | train | 145,215 | `fe81ea26956a2d0c15625992c8604a23194325ce4668595b71102110bfcbcca9` |
| tissue | validation | 31,339 | `092671a60ea271e28db83b5783e07259588d3df826863f273d2967b8031321f2` |
| tissue | test | 31,572 | `5f02c70025e4e1f4b8839460b23bd5575f1dd57ac4a0225e662669cb9b8b310c` |

For each row in each task manifest, resolve its exact
`atlas_row_index,wsi_id,x,y` identity to one and only one
`(run_number,file_index)` in the union. Write the six shared compact location
files under `runs/local/ubc_ocean_latent_store/views/` as
`{task}_{split}_locations.csv` with exact header:

```text
run_number,file_index,atlas_row_index,wsi_id,x,y,split,diagnosis_label,diagnosis_index,tissue_label
```

Rows retain the authoritative task-manifest order. Empty fields remain empty;
integers are decimal; CSV is UTF-8 with LF endings and one header. Each location
file is shared by both models. Pairing one location file with either model store
creates twelve logical views named `{model}/{task}/{split}`. No tensor is copied.

`validate_latent_store_pair` must bind all six task-manifest hashes, reproduce
all six counts and location-file hashes, and additionally prove that exactly
58,666 union rows selected by both tasks resolve to the same physical location
within each model. Public `LatentTaskView` in
`src/eqvae/data/latent_shards.py` takes one model's five shard paths and one
location CSV, opens the correct run mmap, and returns the location row's FP32
tensor plus unchanged task metadata.

## Outputs And Acceptance Artifacts

- `src/eqvae/data/latent_shards.py`: header, manifest binding, resumable writer,
  validation, twelve-view resolution, store-pair validation and mmap dataset.
- `tests/test_latent_shards.py`: focused synthetic contract tests.
- Public exports in `src/eqvae/data/__init__.py` only for the small supported
  writer/reader API.
- No real 73.17 GiB latent artifacts are created locally by this spec.

## Acceptance Criteria

1. Header packing/parsing is exactly 64 bytes and rejects patch magic, wrong
   version, shape, dtype or layout.
2. Synthetic FP32 tensors round-trip byte-exactly through fixed offsets and mmap
   in both sequential and nonsequential row access.
3. Writer accepts only finite CPU FP32 `(B,16,32,32)` batches carrying the exact
   next row start and manifest identities for the next contiguous WSI; swapped
   rows, wrong WSI order, overrun and incomplete commit fail.
4. Resume proves the committed prefix, truncates an incomplete WSI, and produces
   byte-identical final binary/JSON to an uninterrupted write.
5. Header count, file size, CRC32, SHA-256, checkpoint, model, manifest hash and
   JSON completion marker fail closed when mutated. Lightweight open catches
   structural/provenance changes; full artifact validation catches same-size
   payload mutation.
6. JSON is written last and an injected write/finalization failure cannot leave
   a valid completion marker.
7. The reader never shuffles or copies task labels; row `i` remains manifest row
   `i`, and task views can select it through manifest indices.
8. The set validator proves exactly five shards for both models, exact union
   concatenation, trusted provenance, all six task/split counts and hashes,
   twelve model/task/split views, and shared-row offsets.
9. Focused tests, `./scripts/python_quality.sh`, `git diff --check`, repo/workspace
   preflights, and independent clean-context review pass.

## Tests And Verification Commands

```bash
.venv/bin/python -m pytest -q tests/test_latent_shards.py
./scripts/python_quality.sh
./scripts/agent_preflight.sh
```

## Implementation Blockers

None for the implemented storage and twelve-view layer. The ten real shard
payloads are complete remotely. Executing the separately authorized Spec 0021
CPU finalizer to publish the six real location files and global audit remains
pending.

## Verification Result

- The focused storage suite passes 15 tests, including fixed-header rejection,
  exact FP32 round-trip, strict identity/WSI ordering, incomplete-WSI rollback,
  corrupted resume state and payload rejection, all finalization recovery
  windows, sidecar-publication failure, mmap access, provenance mutation, and
  the two-store task-wide proof. The twelve split-view extension is implemented;
  only its real-data Spec 0021 finalizer execution remains pending.
- `./scripts/python_quality.sh` passes: 873 tests passed, one expected GPU-only
  skip, Ruff clean, and BasedPyright with zero errors.
- `git diff --check` passes. No real latent artifact or remote Kaggle action was
  performed.
- Independent clean-context adversarial implementation review found no critical,
  high, medium, or low findings and accepted the implementation against this
  contract.

## Known Risks

- Full FP32 `mu` is about 36.58 GiB per model; each model/run shard is about
  7.2-7.4 GiB. Spec 0021 derives every two-model run's exact binary bytes from
  its frozen row count and rejects it above Kaggle's saved-output cap.
- mmap makes fixed-offset access cheap but cannot make fully random training
  access as cache-friendly as sequential WSI aggregation.
- CRC32 is an accidental-corruption check, not provenance; SHA-256 and the
  checkpoint/manifest bindings carry identity.

## Adversarial Checks

- Mutate one committed payload byte while preserving size.
- Swap manifest rows or use a different shard with the same row count.
- Swap two tensor rows within one WSI while preserving all counts.
- Resume from a state whose last WSI or committed byte boundary is false.
- Append either a full or partial extra uncommitted record and require resume to
  truncate to the derived committed boundary.
- Feed FP16, NaN, wrong shape, a second WSI early, or too many rows.
- Crash after partial-WSI bytes, after final header, after binary rename, and
  before completion JSON publication.
- Open a normal-model bin with SO(2) metadata or a different checkpoint hash.
- Remove/duplicate one run or pair valid shards with a self-consistent but
  untrusted manifest/checkpoint hash.
- Round-trip known FP32 bit patterns and channels-last input to prove explicit
  little-endian CHW serialization; exercise mmap advice fallback and pickling.

## Open Questions

- Spec 0021's fixed smoke recipe and derived saved-output checks must remain
  bound to every physical run.
- The later evaluation spec may consume these frozen views but cannot change
  their physical locations, task/split membership, or hashes.

## Related Files

- `docs/specs/0019-task-consumption-manifests-and-kaggle-work-plan.md`
- `src/eqvae/data/patch_shards.py`
- `src/eqvae/data/dataloaders.py`
- `configs/spec0001/ubc_ocean_masked_holdout_test.json`
