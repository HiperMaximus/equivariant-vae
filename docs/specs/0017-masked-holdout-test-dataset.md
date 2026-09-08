# Spec 0017: Masked-WSI Evaluation Atlas

Status: atlas generated and verified; patch binaries not generated
Owner/workstream: held-out evaluation data
Last updated: 2026-08-24

## Purpose

Create the coordinate atlas from the 152 non-TMA UBC-OCEAN WSIs with
supplemental masks. Keep the generator close to the proven train/validation
notebook, preserve the three mask tissue classes for supervised probes, and
make the atlas independently downloadable so later task manifests or extraction
can resume without recomputing thumbnails or rereading masks. Spec 0018 assigns
these WSIs to classifier train/validation/sealed-test subsets; they remain wholly
unseen during autoencoder training, while primary VAE metrics use Spec 0018's
sealed-test WSIs.

## Non-Goals

- No tissue verifier, class balancing, patch sampling, or shuffling.
- No evaluator, classifier, Kaggle upload, remote run, or paper claim.
- No generic WSI framework or replacement of the train/validation generator.

## Inputs And Data Contract

- Official `train.csv`, `train_images/`, and `train_thumbnails/` from UBC-OCEAN.
- [Official supplemental-mask dataset](https://www.kaggle.com/datasets/sohier/ubc-ovarian-cancer-competition-supplemental-masks);
  filenames select the held-out WSI IDs and the corresponding full-resolution
  RGB images supply incomplete tissue annotations.
- Exactly 152 matching non-TMA rows with labels mapped as
  `CC=0, EC=1, HGSC=2, LGSC=3, MC=4`.
- Full, non-overlapping `256x256` RGB patches.
- Tissue selection is the historical thumbnail policy: Otsu threshold the HSV
  saturation channel and keep a patch when more than `0.6` of its projected
  thumbnail area is foreground.
- Mask annotations are full-resolution RGB: red is tumor, green is stroma
  (healthy tissue), blue is necrosis, and black is unannotated. They are
  non-exhaustive, so black is `unknown`, never a negative tissue label.
- The atlas is additive: retain every full grid patch containing any annotated
  mask pixel, then add every other Otsu-qualified patch.
- The atlas remains lossless with respect to those candidates. Binary extraction
  keeps every Otsu-qualified patch, but a mask-only patch must have at least
  `0.10` annotated coverage. This removes annotation-edge specks without making
  unpainted pixels negative labels or forcing masks/Otsu to be recomputed.

## Outputs And Acceptance Artifacts

- `ubc_ocean_test_atlas.csv`: ordered `wsi_id,label,x,y` rows plus selection
  source, annotation status/label, and total/tumor/stroma/necrosis fractions.
  `mask_status=annotated` means that this patch contains at least one painted
  mask pixel, not merely that its WSI has a mask file. An annotated patch uses
  its largest class fraction as `mask_label`; an exact largest-fraction tie is
  `ambiguous`. An Otsu-only patch is `unannotated/unknown` with zero class
  fractions. The fractions, not the summary label, remain authoritative.
- The default Kaggle artifact path is
  `/kaggle/working/dataset/ubc_ocean_test_atlas.csv`. libvips scratch remains
  outside that publishable tree at `/kaggle/working/tmp_vips`.
- During generation, `ubc_ocean_test_atlas_checkpoint.csv` is the cumulative
  atlas prefix and `ubc_ocean_test_atlas_checkpoint.json` commits the exact byte
  boundary, row count, logical-row SHA-256, and ordered WSI prefix after each
  complete WSI. A
  recoverable Python error after that first commit also writes
  `ubc_ocean_test_atlas_incomplete.json`; the final atlas remains absent so the
  partial output cannot be mistaken for completion. A successful run promotes
  the checkpoint CSV to the final atlas and removes both JSON files.
- Five `ubc_ocean_test_part_XX_of_05.{bin,csv,json}` groups. Each part contains
  complete, contiguous WSIs and is a valid CHW `uint8` `UBC_DATA` shard with
  its own CRC32 and provenance. A failed Kaggle session therefore loses at
  most one part, not the whole extraction.
- `merge` recreates `ubc_ocean_test.{bin,csv}` and
  `ubc_ocean_test_provenance.json` by concatenating the five parts in number
  order, validating them against the atlas, and recomputing global checksums.
- One readable implementation: `kaggle/generate_ubc_ocean_test.py`.

## Workflow Contract

1. `atlas` stage sorts selected WSIs numerically, opens each WSI only to read its
   true dimensions, thresholds its thumbnail, and streams the corresponding
   full-resolution mask in 256-pixel-high rows. Nonblack mask pixels use their
   dominant RGB channel so antialiased edges retain their class. If two or three
   channels tie for the maximum, that pixel contributes to every tied class;
   `annotated_fraction` still counts the pixel once. Class fractions may
   therefore overlap slightly and need not sum exactly to annotated coverage.
2. The selected coordinates are the deduplicated union of all mask-intersecting
   grid patches and all Otsu-qualified patches, written in `wsi_id,y,x` order.
   After each complete WSI, the CSV is flushed and fsynced before an atomic JSON
   state update commits its byte boundary. Resume truncates any uncommitted next-
   WSI tail, validates every retained row and the exact WSI/label prefix, then
   verifies the logical-row SHA-256 and continues with the first unfinished WSI.
3. The atlas is complete input to the `dataset` stage and can be downloaded and
   remounted in a later Kaggle session. `--resume-atlas-checkpoint-dir` accepts
   an explicit attached output; when omitted, exactly one complete checkpoint
   pair below `/kaggle/input` is discovered by its stable filenames. Missing,
   half-paired, or multiple checkpoints fail instead of being guessed. The
   guarded kernel permits the exact private
   `maximusshtefan/eqvae-ubc-ocean-test-atlas-checkpoint` dataset as its sole
   optional resume source; arbitrary extra sources remain forbidden.
4. Extraction eligibility is `passes_otsu OR annotated_fraction >= 0.10`.
   The atlas keeps lower-coverage mask-only candidates for audit, but part plans,
   projected binary size, part CSVs, and the merge count only eligible rows.
5. Eligible atlas patch counts divide the ordered WSI list into five contiguous,
   approximately patch-balanced parts. WSIs are never split: every extraction
   run opens only the WSI range assigned to `--part`.
6. `dataset --part N` preserves atlas order and all annotation columns inside
   part `N`. It disables the
   unhelpful global libvips operation cache, opens one WSI at a time with
   sequential access, groups eligible rows by `y`, and fetches one 256-pixel-high
   horizontal span from the first through last selected `x` on that row. Patches
   are sliced and written one at a time, then the span is released before the
   next `y`; no multi-row or WSI-sized pixel buffer exists.
7. `merge` requires every expected `.bin`, `.csv`, and `.json`. It verifies the
   provenance atlas hash, part number, exact WSI list, patch count, CRC, and
   full binary SHA-256 before accepting the named part, then streams payloads
   into the final binary. It does not decode images or rerun Otsu.
8. Bounds, RGB channel count, atlas order, labels, mask fractions/status,
   header fields, patch count,
   and checksums fail closed. No patch is silently skipped.
9. The default `atlas` stage stops after printing candidate/eligible totals and
   per-part patch
   counts and projected GiB. The script writes under
   `/kaggle/working/dataset` by default and accepts explicit atlas, output, and
   part-input directories.
10. The CPU-only, Internet-enabled script kernel at
    `kaggle/kernels/ubc_ocean_test_atlas` attaches only `UBC-OCEAN` and the
    official supplemental-mask dataset for a fresh run; a resume may add only
    the exact private checkpoint dataset named above. Its generated uploadable
    `run.py` is a byte-for-byte copy of the readable generator. On a real Kaggle
    worker the generator resolves the legacy direct and current
    typed/owner/versioned source mounts by exact identity and required-file
    signatures, sets writable scratch, then installs the historical libvips
    dependency and pinned `pyvips==3.1.0` only when import is missing. Explicit
    input paths remain strict overrides.
11. Kaggle version 2 exposed no downloadable files after its Python exception.
    Therefore, when atlas generation raises after at least one committed WSI,
    the script first revalidates the full committed prefix, then prints the
    traceback, writes the explicit incomplete marker, and returns successfully
    so Kaggle can publish the checkpoint pair. Invalid attached checkpoints,
    errors before the first complete WSI, and pre-existing final atlases fail
    normally. This protects ordinary Python failures, not an OOM, machine loss,
    or forced process termination.

## Acceptance Criteria

1. Mask filenames select exactly the canonical 152 non-TMA WSIs.
2. Atlas coordinates equal the mask-or-Otsu union and contain only full
   in-bounds patches. Every mask-intersecting patch is retained in the atlas;
   only mask-only candidates below `0.10` coverage are omitted from binaries.
3. A saved atlas can drive any of five extraction runs without masks or
   thumbnails; rerunning one part produces identical content.
   A checkpointed atlas run resumes at the first unfinished WSI without opening
   or thresholding completed WSIs, and ignores an uncommitted CSV tail.
4. Every part and the merge are valid standalone shards. Merged binary row
   order equals the eligible atlas subsequence (`passes_otsu OR mask-only
   annotated_fraction >= 0.10`) and merged CSV order, and validates with the
   existing `PatchShard` reader, including CRC32.
5. Red/green/blue are read as tumor/stroma/necrosis, blended boundary pixels
   retain their dominant class, maximum-channel ties contribute to every tied
   class, and unannotated pixels remain unknown. Patch labels use the largest
   class fraction and reserve `ambiguous` for an exact patch-level tie.
6. No tissue verifier, randomization, class balancing, multiprocessing, or
   multi-row patch batching exists in the implementation.
7. The guarded Kaggle build reproduces `run.py` from the generator, validates
   exact CPU/Internet/source metadata, and refuses a stale upload file.
8. Focused tests, `./scripts/python_quality.sh`, and
   `./scripts/agent_preflight.sh` pass.

## Known Risks

- Kaggle kernel version 1 proved dependency installation but failed before
  atlas work because it assumed the historical direct mask mount. Version 2
  resolved the current mount and completed 78 WSIs before the old unique-
  dominant-color guard rejected two tied red/green boundary pixels in WSI
  28562. Its failed status exposed no downloadable checkpoint. Version 3
  implements the accepted overlap rule and whole-WSI checkpoint publication;
  it completed all 152 WSIs and its downloaded final atlas validates locally.
  The launcher still fails before atlas output if the pinned pyvips binding
  cannot load libvips.
- Keeping every Otsu-selected region may produce a large binary. The atlas row
  count and estimated binary size must be printed before extraction; selection
  must not be changed after model outputs are inspected. Version 3 measured
  1,816,752 eligible patches and 332.657 GiB raw CHW payload. The original five
  parts project to 65.301--67.646 GiB each, above Kaggle's normal saved-output
  limit, so those five extraction runs are blocked pending a revised transport
  or sealed sampling design.
- Supplemental masks are non-exhaustive. Unannotated WSI regions are not
  negative labels and must not enter the tissue-classification loss as stroma.
- The local mechanics tests use a fake pyvips backend. Real atlas version 3
  completed all 152 WSIs and its downloaded CSV passes the same cohort, order,
  fraction, eligibility, and part-plan validators. Real patch extraction remains
  separately authorized and is currently blocked by projected output size.
- Whole-WSI checkpoints cover Python exceptions that reach the launcher. They
  cannot publish themselves after OOM, worker loss, or a hard Kaggle timeout;
  the normal atlas is currently projected to finish far below the CPU limit,
  based on version 2 processing 78 WSIs in about 66 minutes.
- Part balancing uses eligible atlas patch count as a work estimate. One unusually large
  WSI can still dominate a part because splitting one WSI across sessions would
  sacrifice the cleanest restart boundary.

## Adversarial Checks

- Change atlas order, one label, one coordinate, payload bytes, or header count
  and confirm the reader/test fails.
- Confirm the dataset stage works with the thumbnail and mask paths absent.
- Confirm a TMA row or a mask ID absent from `train.csv` fails selection.
- Remove, reorder, or alter one part and confirm merge fails before publishing
  the final shard.

## Related Files

- `configs/spec0001/ubc_ocean_masked_holdout_test.json`
- `docs/data/ubc_ocean_masked_holdout_ids.csv`
- `docs/behavior_inventory_kaggle.md`
- `src/eqvae/data/patch_shards.py`
