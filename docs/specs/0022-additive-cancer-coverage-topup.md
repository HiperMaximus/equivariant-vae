# Spec 0022: Additive Cancer Coverage Top-Up And WSI Bag Manifests

Status: quarter-coverage top-up complete; single-WSI completion extension locked
Implementation readiness: contract accepted after independent clean-context
Terra review; independent adversarial implementation re-review passed
Owner/workstream: held-out latent dataset completion
Last updated: 2026-08-29

## Purpose

Replace the fixed 3,000-patch cancer cap with a per-WSI coverage rule suitable
for attention MIL, without regenerating the 599,398-row base latent store.
Create one additive, dual-model top-up shard and one model-independent logical
WSI dataset whose exact patch membership is fixed before classifier training.

For WSI `w`, let `A_w` be its complete mask-independent Otsu candidate count:

```text
K_w = min(A_w, max(1000, ceil(0.25 * A_w)))
```

Implement the quarter term with integer arithmetic as `(A_w + 3) // 4`; no
floating-point rounding participates in dataset membership.

Every WSI classifier bag contains exactly `K_w` coordinates. The same bags,
order, labels, and split are consumed by the normal and continuous-`SO(2)`
latent stores.

## Single-WSI 45630 Completion Extension

Locked / implementation-ready. The user requests one extra latent part for the
missing foreground patches of the largest training WSI. This does not change
the completed quarter-coverage dataset or authorize classifier training. The
user clarified that the full bag is only for a subsequent memory-fit check,
not a learning experiment or campaign-wide coverage change.

- Only WSI `45630`, verified `train`, is in scope. Reuse the hash-bound Spec
  0018 Otsu candidate manifest derived from the atlas. Filter this WSI before
  interpreting rows; do not open validation/test logical files, pixels, or
  binaries. All 32,595 full-grid Otsu coordinates are the target, independent
  of masks or model outputs. The atlas's 57 mask-only coordinates are excluded.
- Compute the missing set against the entire original physical union and
  completed part-11 manifest, not just the current 8,149-row MIL bag. Verify
  atlas-row identity and coordinates agree wherever sets intersect. Expected
  reuse is 5,136 Otsu rows in base part 4 and 4,810 in part 11, leaving exactly
  **22,649** rows, ordered `y,x`. Existing mask-only base rows are not targets.
  Authenticate candidate, union, part-11 manifest, completed logical audit and
  source top-up contract against their existing sealed hashes before deriving.
- Stage only the missing manifest (`atlas_row_index,wsi_id,x,y`), the two
  frozen step-60000 checkpoints, and a new top-up input contract. Copy the
  existing `src/eqvae` source into a private input bundle as a sibling of the
  four-file `inference/` directory; pass only `inference/` to the unchanged
  worker. Hash/allow-list the source before importing it. Never embed that
  source or the manifest/checkpoints into `run.py`. Input dataset slug is
  `maximusshtefan/eqvae-wsi45630-completion-inputs`, intended immutable v1.
  Local manifest/audit generation and package staging are authorized. Private
  dataset publication and kernel launch require separate user authorization;
  launch requires a byte-verified private version-1 input receipt.
- Reuse `generate_ubc_cancer_topup.run_topup` unmodified, with its FP32,
  batch-8, clean-RGB-to-posterior-mu path, one VAE per T4 and shared patch reads.
  Reuse the existing writer, checkpoint hashes, row alignment, per-WSI commit,
  sidecars and pair-audit-last promotion. Generate no raw RGB dataset, pooled
  vectors, masks, predictions or model training. Extraction batches are not
  MIL bag chunking; the future classifier still consumes the full bag.
- Use a distinct private kernel `maximusshtefan/eqvae-wsi45630-completion`,
  attaching only its new input dataset and official `UBC-OCEAN` raw source.
  The worker must read only `45630.png`; no existing latent output needs to be
  mounted. Preserve established input resolution, latest-Torch-before-import
  and pyvips setup. Record Torch/CUDA versions and compact completion status
  in the Kaggle log; keep the existing five-file publishable output unchanged.
- Preserve worker filenames `normal_vae_mu_cancer_topup.bin`,
  `so2_vae_mu_cancer_topup.bin`, their sidecars and pair audit, but under the
  distinct kernel's `dataset/` output. Each binary is **1,484,324,928 bytes**;
  pair total **2,968,649,856 bytes**, plus small metadata, below 20 GB. They
  are new files, never replacement versions of the existing part-11 kernel.
  The later catalog may assign logical part 12 after completion verification;
  do not modify live training manifests/catalogs or claim a full bag exists yet.
- Recalculate the existing deadline projection for 22,649 rows and one WSI
  from the sealed original run timing summary, retaining the 3,600-second
  reserve and 28,800-second session cap. The reused worker consumes the
  historical `worst_observed_seconds_per_wsi` key as its start bound: set it
  to the larger row/WSI projection, **1,960 seconds**, not the old 353-second
  average-slide value. Total planning allowance is **5,560 seconds**. This is
  a planning estimate, not a
  guaranteed wall time. No new runtime search or generalized resume framework.
- Acceptance: exact missing-set/identity/count checks; tiny synthetic tests of
  subtraction including already-stored-but-unselected rows and wrong-split
  rejection; input/runtime-contract and source-size checks; focused worker
  reuse test; independent clean-context adversarial review and repo Python
  quality gate. Never generate/download a real `.bin` locally. Remote success
  later requires one completed WSI, exactly 22,649 aligned records per model,
  both sidecar/payload hashes, and the complete pair audit. Download compact
  evidence only, leaving payloads remote.

## Non-Goals

- No change to the frozen 106/23/23 WSI split, VAE checkpoints, latent shape,
  five base work manifests, ten base binaries, or hash-bound Spec 0021.
- No physical train/validation/test binary copies and no per-WSI latent files.
- No tissue-task resampling or tissue latent top-up.
- No classifier, attention model, metric, test prediction, paper claim, or
  public derived-data release.
- No reusable inference platform, arbitrary hardware support, alternate model
  pair, general shard format, or production-service behavior. This is one
  audited run on the existing Kaggle dual-T4 setup.

## Inputs And Data Contract

- Complete cancer candidate manifest from Spec 0018:
  `runs/local/ubc_ocean_eval_manifests/cancer_ae_patch_manifest.csv`, exactly
  1,750,221 mask-independent Otsu rows, SHA-256
  `710c3f8166f577f5ae60bec94a544dabed9619342a46b82374a2e739162d648c`.
- Existing Spec 0019 cancer selections:
  `runs/local/ubc_ocean_eval_consumption/cancer_{train,validation,test}.csv`.
- Existing physical union:
  `runs/local/ubc_ocean_eval_consumption/union_patch_manifest.csv`, exactly
  599,398 rows, SHA-256
  `f92558fa7aced13debc839c733e2c96d03c1a3df4b2a0b194b60558c69c04012`.
- Frozen WSI split:
  `docs/data/ubc_ocean_eval_wsi_split.csv`, exactly 106 train, 23 validation,
  and 23 sealed-test WSIs, SHA-256
  `216f69f64ed7a3e5636173d6cfe83297632113bc68310e4f6285d7ffc22cd43c`.
- Five validated normal and five validated SO(2) base shards plus the completed
  Spec 0021 global audit. The global audit is an execution dependency; this
  spec never weakens or replaces it.
- Exact frozen checkpoints and FP32 posterior-`mu[16,32,32]` contract from
  Specs 0020-0021.
- Official UBC-OCEAN WSI PNGs. Source patches remain the fixed, non-overlapping
  256-by-256 grid identified by `(atlas_row_index,wsi_id,x,y)`.

Rows in every selection input are authoritative and sorted by `wsi_id,y,x`.
Labels and split come only from the frozen WSI assignment. Model outputs,
supplemental mask membership, and tissue selection may not influence cancer
coordinate membership.

## Stable Selection Contract

Use base selection seed `20260827`. For each WSI, construct independent NumPy
`PCG64(SeedSequence([20260827,wsi_id,purpose]))` streams over authoritative
sorted row positions. `purpose=0` selects a smaller retained set and
`purpose=1` selects additions. Sampling is without replacement; selected rows
are restored to `y,x` order before publication.

1. Compute `K_w` from the complete Spec 0018 candidate count.
2. If the existing cancer selection has fewer than `K_w` rows, retain every
   existing row and select exactly the deficit from the remaining candidates.
3. If it has more than `K_w` rows, select exactly `K_w` rows from the existing
   selection. No new coordinate is generated for that WSI.
4. If counts are equal, retain the existing selection byte-for-byte.
5. Freeze the complete target set `T` before resolving physical storage.
6. Resolve `T` against the complete base physical union `B`, not only the old
   cancer selection. Reuse `T intersect B`, including coordinates stored for
   the tissue task. Encode only `T minus B`.
7. Tissue membership may reduce physical work only after `T` is frozen; it may
   never change which cancer coordinates enter `T`.

The reviewed count projection is:

| Split | Target rows | Nominal additions | Under-covered WSIs | Old rows omitted |
| --- | ---: | ---: | ---: | ---: |
| train | 308,359 | 56,699 | 43 | 63,095 |
| validation | 66,706 | 12,770 | 11 | 14,109 |
| test | 66,194 | 10,444 | 11 | 11,388 |
| total | 441,259 | 79,913 | 65 | 88,592 |

`79,913` is an upper bound on new inference because some selected additions may
already be tissue-only rows in `B`. The exact supplement count is derived and
sealed before the remote run.

The implemented read-only local derivation resolves 367,226 target rows to the
base and leaves 74,033 rows across 65 WSIs for the supplement. The exact paired
binary size is 9,703,653,504 bytes (9.037 GiB), leaving 10,296,346,496 bytes
before Kaggle's cap. These values are not sealed execution authority until the
Spec 0021 global audit exists.

Target bag sizes have minimum 1,000, median 2,766.5, and maximum 8,149 rows.

## Logical Dataset Contract

Write one instance CSV and one bag-index CSV per split:

```text
wsi_cancer_train_instances.csv
wsi_cancer_train_bags.csv
wsi_cancer_validation_instances.csv
wsi_cancer_validation_bags.csv
wsi_cancer_test_instances.csv
wsi_cancer_test_bags.csv
```

Instance header:

```text
instance_row,atlas_row_index,wsi_id,x,y,diagnosis_label,diagnosis_index,split,store_source,shard_number,file_index
```

`store_source` is exactly `base` or `topup`. Base rows use shard numbers 1-5;
top-up rows use shard number 1. `file_index` is the zero-based record in the
selected physical shard. Instance files are ordered `wsi_id,y,x`; their
`instance_row` is the zero-based CSV data-row position.

Bag header:

```text
bag_row,wsi_id,diagnosis_label,diagnosis_index,split,instance_start,instance_count
```

Each bag is one contiguous instance range and contains exactly `K_w` rows.
Bag files are derived from completed instance files and are never an
independent selection authority. The same instance and bag CSVs are paired
with either model store; no model-specific logical CSV is allowed.

## Top-Up Inference Contract

- Run exactly one private Kaggle GPU job with exactly two visible Tesla T4s.
- The mounted inference manifest contains only
  `atlas_row_index,wsi_id,x,y`, in exact supplement order. It contains no
  diagnosis, diagnosis index, split, tissue label, or mask field. Logical
  labelled instance/bag CSVs are built and audited outside the GPU kernel and
  are not attached to it.
- Reuse the Spec 0021 clean-RGB normalization, checkpoint loading, posterior
  `mu`, FP32 serialization, per-WSI ordering, sidecars, and pair-audit-last
  mechanics. Do not refactor them into a generic platform.
- Sort the supplement manifest by `wsi_id,y,x`, open one WSI at a time, read
  each missing RGB patch once, and send the same batch to normal VAE on
  `cuda:0` and SO(2) VAE on `cuda:1`.
- Emit exactly one aligned pair:
  `normal_vae_mu_cancer_topup.bin` and
  `so2_vae_mu_cancer_topup.bin`, their completion sidecars, and one pair audit.
- The two binaries have the same row count and coordinate order. Row `i` in
  both files must represent the same manifest identity.
- Do not write RGB patches, reconstructions, `z`, `logvar`, predictions, or
  metrics.
- Derive the exact saved bytes as `2 * (64 + rows * 65,536)` and fail before
  inference unless the binaries plus a 10 MB metadata reserve fit Kaggle's
  20,000,000,000-byte saved-output cap. The nominal upper bound is about
  9.76 GiB for both binaries combined.
- Retain only correctness protection needed for this one run: exact input
  hashes, complete-WSI commits, aligned pair completion, and audit-last
  publication. No CPU fallback, alternate GPU count, model discovery, or
  compatibility layer is required.

## Outputs And Acceptance Artifacts

- Readable manifest/top-up builder and focused tests.
- One immutable compact top-up inference contract containing only the
  label-free supplement manifest, checkpoint/model bindings, input hashes, and
  exact output-size projection.
- One separate compact logical-dataset contract containing the labelled target
  instance/bag CSVs. It is not mounted in the inference kernel.
- One dual-model top-up binary pair with completion JSON and pair audit.
- The complete five-file pair directory is promoted atomically only after both
  payloads validate and the pair audit has been written last; no audit can
  publish separately from an incomplete pair.
- One logical-dataset audit binding all six logical CSVs, the frozen split,
  complete target selection, base global audit, top-up pair audit, model
  checkpoints, exact counts, and file hashes.
- One compact access transcript schema for later classifier runs so train and
  validation reads can be proved not to include test locations.
- No paper, issue, thesis, GitHub, public-dataset, or remote action without a
  separate explicit request and authorization.

## Acceptance Criteria

1. `K_w` is reproduced exactly for all 152 WSIs from the complete cancer
   manifest; target totals equal 308,359/66,706/66,194 and 441,259 overall.
2. Every target row comes from the mask-independent candidate manifest, has
   one frozen split, and appears exactly once. No WSI or coordinate crosses
   splits.
3. Under-covered WSIs retain every old cancer row; over-covered WSIs use only
   old cancer rows; all membership is byte-stable across repeated generation.
4. The target is frozen before base-union resolution. Every target coordinate
   resolves exactly once to either base or top-up, never both or neither.
5. No supplement coordinate exists in the base union. Because candidates are
   on the fixed 256 grid, this also proves no regenerated or spatially
   overlapping patch.
6. Per-WSI 10-by-10 occupied-cell coverage is recorded before and after the
   revision, together with candidate/selected counts. The audit exposes any
   loss of broad tissue coverage rather than hiding it behind aggregate counts.
7. Every bag range is contiguous, gap-free, non-overlapping, label-consistent,
   split-consistent, and exactly equal to that WSI's instance rows and `K_w`.
8. Both top-up binaries validate fully and are row-aligned to the supplement
   manifest. Their sidecars bind the correct model/checkpoint pair and the pair
   audit is written last.
9. Pairing the same logical CSVs with either complete model store resolves the
   same ordered coordinate identities. Model choice changes only latent bytes.
10. The test logical files may be created and the label-free top-up may encode
    test rows, but no classifier may open test locations before Spec 0023's
    protocol and training outputs are locked.
11. The top-up kernel input tree contains no diagnosis, tissue, mask, or split
    field; a test label cannot be read by the label-free inference process.
12. Focused tests, `git diff --check`, repo/workspace preflights, and an
    independent clean-context adversarial review pass before implementation is
    called ready. Python implementation later requires the repo quality gate.

## Tests And Verification Commands

Implemented commands:

```bash
.venv/bin/python -m pytest -q tests/test_spec0022_cancer_topup.py
.venv/bin/python -m eqvae.cli.build_ubc_cancer_topup --validate-only
./scripts/kaggle_kernel.sh build-cancer-topup
./scripts/kaggle_kernel.sh preflight-cancer-topup
./scripts/kaggle_kernel.sh publish-cancer-topup-inputs
./scripts/kaggle_kernel.sh verify-cancer-topup-inputs
git diff --check
./scripts/agent_preflight.sh
```

The focused suite must cover exact `K_w`, deterministic independent RNG
streams, under/over/equal cases, target-before-union independence, tissue-only
reuse, duplicate/overlap rejection, WSI split leakage, missing/ambiguous
physical resolution, bag ranges, pair alignment, hashes, saved-output limits,
and audit-last failure.

## Implementation Blockers

None for local implementation. Execution still requires the completed and
validated Spec 0021 CPU finalizer as an immutable base dependency. Any Kaggle
build, upload, or run requires a later exact user authorization.

## Adversarial Review Result

The independent Terra contract review found no leakage defect in
target-before-union selection or physically mixed shard use. It required the
GPU input to become a strictly label-free coordinate manifest and the logical
labelled CSVs to remain separate. The independent adversarial implementation
review then required full logical-CSV revalidation, atomic paired output
publication, exact wrapper-byte regeneration, per-WSI deadline budgeting, and
a remotely verified private-input receipt. Those findings are implemented;
the clean-context re-review found no remaining scientific, correctness, or
launch-gate blocker.

## Known Risks

- The 25% rule controls sampled tissue-area coverage, not biological diversity
  or independent sample size. The WSI remains the statistical unit.
- Randomly reducing an over-covered systematic selection can lose rare spatial
  regions. The per-WSI spatial audit makes this visible; a failed coverage
  review must be resolved before the manifest is sealed.
- PNG decode time depends on 65 WSIs and their dimensions, not only supplement
  rows. Output size fits one job, but the exact preflight must also project the
  one-run deadline from the completed Spec 0021 evidence.
- The physical base is mixed across splits. Leakage protection therefore lives
  in immutable logical CSVs, loader access transcripts, and WSI-level audits,
  not physical file separation.

## Adversarial Checks

- Select additions after consulting tissue membership and require failure.
- Put a base-union coordinate in the supplement manifest and require failure.
- Move one WSI patch to another split while preserving global counts.
- Swap two top-up tensor rows or pair different model manifests.
- Change a bag range so it overlaps, skips, or partially consumes a WSI.
- Preserve total rows while violating one WSI's `K_w`.
- Use different logical CSVs for the two models or dynamically resample a bag.
- Allow a training access transcript to contain any test physical location.
- Mount a labelled logical CSV in the top-up inference job.

## Open Questions

None for the scientific data contract. The local projection uses all five
completed Spec 0021 runs and conservatively takes the worse of observed
seconds per row and seconds per WSI: 22,884 inference seconds plus the locked
3,600-second reserve, or 26,484 seconds within the 28,800-second session. The
manifest, artifact hashes, and projection must still be sealed after the base
global audit exists and before the one remote run.

## Related Files

- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0019-task-consumption-manifests-and-kaggle-work-plan.md`
- `docs/specs/0020-fp32-latent-shard-format-and-io.md`
- `docs/specs/0021-dual-model-wsi-latent-inference.md`
- `src/eqvae/data/latent_shards.py`
- `src/eqvae/inference/dual_writer.py`
- `src/eqvae/cli/build_ubc_cancer_topup.py`
- `src/eqvae/cli/generate_ubc_cancer_topup.py`
- `src/eqvae/cli/build_ubc_cancer_topup_kernel.py`
- `kaggle/kernels/ubc_ocean_cancer_topup/run_template.py`
- `tests/test_spec0022_cancer_topup.py`
