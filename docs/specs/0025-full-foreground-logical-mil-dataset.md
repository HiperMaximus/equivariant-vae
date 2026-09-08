# Spec 0025: Full-Foreground Logical MIL Dataset

Status: implemented / locally verified
Owner/workstream: reference-only full-coverage supervised data integration
Last updated: 2026-08-31

## Scope And Boundaries

Unify the 15 verified physical part pairs into full-foreground MIL views. This
is a local metadata/reader change, not extraction, training, publication or test
release. Preserve all existing quarter-coverage MIL and tissue artifacts at
`runs/local/ubc_ocean_supervised_manifests` byte-for-byte. Do not edit sealed
Specs 0021/0023/0024, their source snapshots, or executed packages. This separate
contract implements Spec 0024's pending logical integration.

No binary downloads, copies, repacking, image reads, test-feature/label loading,
new architecture, tuner, Kaggle launch, paper or remote write. Test coordinate
metadata may be joined for coverage and reserved mappings, never for selection.
No training protocol, augmentation, metric, seed or schedule changes.

## Inputs And Physical Resolution

Authenticate the completed aggregate audit (SHA-256
`5f21633d42cce71955ba3f6f0cf9c2dbd5a6d4b5b19b8339e745775d1fe76aae`)
and independent atlas coverage audit (SHA-256
`6eb4a0de6c012a56b1eaf5c42c553eec2ece28293257c0d0b172ba1c4d8c7139`).
Follow their pinned plan, candidate/split hashes, old completion audits and
per-run manifest/sidecar hashes. These are completed immutable authorities;
do not rerun extraction-package validation against today's changed source tree.

Use the existing `physical_parts.csv` schema and catalog reader:

- Parts 1–5: original five producers; 11: ordinary first top-up.
- Part 12: WSI45630 supplement; parts 13–20: full-foreground jobs 01–08.
- Every part has exactly one normal and one SO(2) row. Bind each to the exact
  producer (all completed version 1), filename, bytes, binary hash, sidecar
  hash and authenticated source manifest. Reused basenames and legacy worker
  run number 1 must never resolve a part on their own.
- `file_index` is the zero-based position in that physical part's original
  manifest, NOT the position after foreground filtering. Retain physical gaps.
- Both branches use one model-independent `(part,file_index)` mapping and
  identical `(atlas_row_index,wsi_id,x,y)` identities. Shape/dtype stay FP32
  posterior `mu[16,32,32]`; binary records remain 64-byte header + 65536 bytes/row.

Stream a numeric `wsi_id,y,x` merge of all authenticated stored manifests and
the fixed Otsu candidate manifest. Reject overlap, identity drift, missing
targets and unplanned extra supplement rows. Only the 23,023 already-stored
mask-only base rows are excluded. Cross-check all 152 per-WSI foreground counts
against the independent atlas audit, not only totals. Preserve the frozen
106/23/23 WSI assignment; labels for development come only from train/validation
rows of the frozen split. Do not consume candidate/test diagnosis fields.

## Outputs And Development Access

Default output: `runs/local/ubc_ocean_full_foreground_manifests`. Refuse overwrite;
publish the completed local directory only after validation.

- `development/physical_parts.csv`: 30 catalog rows.
- `development/wsi_cancer_{train,validation}_{instances,bags}.csv`: existing
  Spec 0023 schemas, complete bags in numeric coordinate order.
- `development/dataset.json`: exact five-file allow-list, byte counts/hashes,
  train/validation per-WSI bag identities/counts and source-audit identity.
- `sealed_test/`: label-free test location and bag-range CSVs only. No test
  diagnosis or class index. These mappings are NOT development inputs.
- Root integration audit: input/source provenance, all output hashes, per-WSI
  coverage, split counts, excluded base rows and explicit no-test-release state.

The development loader requires an externally pinned SHA-256 of `dataset.json`,
accepts only `train` or `validation`, validates the exact development file set
and hashes plus requested split/WSI membership before opening any binary, then
reuses `SupervisedLatentStore` and complete `WSIBagDataset` read behavior. Read
by physical part/increasing offset and restore logical order. No chunking,
truncation, random sampling or per-model membership. Catalog roots refer to
`/kaggle/input/notebooks/maximusshtefan/<producer>/dataset` when mounted.

Mixed-split physical files necessarily remain mounted; this is a controlled
logical-access boundary, not OS-level isolation from arbitrary malicious code.
Future learning packages must stage ONLY `development/` metadata and pin its
contract hash. They must preserve paired access transcripts and the Spec 0023
release gate. No training integration or release loader is enabled here.

## Acceptance And Verification

- Exact foreground rows/model: train 1,224,875; validation 264,178; reserved
  test 261,168; total 1,750,221. All 152 bags exactly match the independent audit.
- Unique source resolution and paired coordinates across all 15 parts; no
  missing row, duplicate or split-crossing WSI; largest bag 32,595 is intact.
- Focused synthetic tests: repeated source filenames, physical gaps/restored
  order and complete paired bags, duplicate/missing/drift rejection, split
  mismatch/test refusal before physical reads, altered metadata/sidecar refusal.
- Run the real metadata-only build and audit old-output/sealed-spec preservation.
- Run `./scripts/python_quality.sh` with focused pytest selection, full lint/type
  checks, `git diff --check`, preflight and clean-context adversarial review.
- No dataset-dependent hyperparameter or learning claims follow from this work.

## Risks, Review And Related Contracts

Review source identity versus reused basenames, off-by-one offsets after
filtering, paired mismatch, split leakage, accidental bag truncation, protection
of frozen tissue/quarter views and needless abstractions. No implementation
blockers remain: user authorized local integration, all 15 pairs are complete.
Related: Specs 0018–0024, `CURRENT.md`, `GOAL.md`,
`docs/repo_goal_and_requirements.md`, `docs/agentic_review_workflow.md`.
No paper/issue/image deliverable changes.

## Usage And Verified Result

Builder: `src/eqvae/cli/build_ubc_full_foreground_manifests.py`.
Reader: `src/eqvae/data/full_foreground_latents.py`.
The metadata-only command is:

```bash
.venv/bin/python -m eqvae.cli.build_ubc_full_foreground_manifests
```

The default output already exists and is accepted; this command deliberately
refuses to overwrite it. Stage only the six files inside `development/`, never
the enclosing output directory. This reader is not yet wired into a new
transformer learning runner or Kaggle package.

### How To Load A WSI: Agent Recipe

The logical dataset is an address book for existing **patch embeddings**, not
RGB patches or a new binary. The lookup chain is:

```text
WSI ID in the requested split
  -> bag row: instance_start, instance_count
  -> that slice of the instance CSV: (part, file_index) for every patch
  -> catalog[(part, model_name)]: Kaggle producer + binary filename
  -> byte offset 64 + file_index * 65536 inside that binary
```

All indices are zero-based data-row indices; CSV headers are not rows.
`instance_start` indexes the logical instance CSV, while `file_index` indexes
the original physical binary. They are NOT interchangeable. One binary record
is one FP32 `mu[16,32,32]`. Part IDs identify source pairs, not splits or WSIs.

Worked metadata example: training WSI `45630` has `bag_row=77`,
`instance_start=836492`, `instance_count=32595`, so its logical range is
`instances[836492:869087]`. Its embeddings are distributed as follows:

| Part | Foreground embeddings | Producer under `maximusshtefan/` |
| --- | ---: | --- |
| 4 | 5,136 | `eqvae-ubc-ocean-latent-run-04` |
| 11 | 4,810 | `eqvae-ubc-ocean-cancer-latent-top-up` |
| 12 | 22,649 | `eqvae-wsi45630-completion` |

For example, logical instance row `836493` points to `(part=11,file_index=48277)`.
Never infer physical indices from bag counts or renumber them after filtering.
Parts 11 and 12 reuse identical filenames but have different producer
directories and sidecar hashes. Normal/SO(2) use identical logical pointers;
only the catalog's `model_name` selection changes the binary.

The following is usage inside a future authorized Kaggle package, not a local
payload-read command or launch authorization. `dev_root` must point to that
package's staged development directory; `approved_contract_sha256` comes from
its externally pinned config (accepted value in `CURRENT.md`), never a digest
computed from whatever JSON happens to be mounted.

```python
import csv
from pathlib import Path

from eqvae.data.full_foreground_latents import FullForegroundBagDataset

with (dev_root / "physical_parts.csv").open(newline="", encoding="utf-8") as handle:
    sources = {row["kaggle_source"] for row in csv.DictReader(handle)}
source_roots = {
    source: Path("/kaggle/input/notebooks") / source / "dataset" for source in sources
}

with FullForegroundBagDataset(
    development_root=dev_root,
    expected_contract_sha256=approved_contract_sha256,
    split="train",
    model_name="normal_vae",  # Separate paired instance uses "so2_vae".
    source_roots=source_roots,
) as dataset:
    by_wsi = {bag.wsi_id: index for index, bag in enumerate(dataset.bags)}
    loaded = dataset[by_wsi[45630]]  # dataset[45630] would be WRONG.
    latents = loaded.latents  # FP32 CPU tensor: [32595, 16, 32, 32].
    identities = loaded.instances  # Same ordered identities for either model.
```

Construct once per model/split and reuse across WSI reads, not once per WSI.
Attach all 15 distinct catalog producers as `kernel_sources`: the current
store validates sizes/headers/sidecars for the full model catalog at creation,
then maps payloads lazily for the requested parts. Reading this WSI needs
payload rows from only parts 4, 11 and 12. Do not prune the sealed catalog or
discover binaries by basename/glob. The root map assumes the established
Kaggle notebook-output mount layout; a missing source is a mount error, not
permission to skip its patches or download them locally.

The reader groups requested records by part/increasing physical offset and
copies them into one in-memory bag in logical `wsi_id,y,x` order. Gaps in old
parts stay gaps; it does not read every intervening record, create a combined
`.bin`, pool, truncate or subsample. This temporary tensor assembly is distinct
from forbidden on-disk binary copies. A missing WSI in `by_wsi` must fail;
never fall back to another split. Use `split="validation"` explicitly for
validation. `split="test"` is rejected; never use the lower-level store to
bypass this boundary. Physical shared shards are not OS-level split isolation.

Real local build and independent reverse-pointer verification pass: each of
1,750,221 output identities resolves to the original manifest record at its
declared part/index, and the merged logical identities equal the frozen
candidate manifest exactly. All 152 per-WSI counts match; the original catalog's
12 rows remain the exact prefix of the new 30-row catalog. Fifteen new tests and
15 existing manifest/reader/model tests pass through the focused repository
quality gate; whole-repo Ruff/BasedPyright clean, `GATE_EXIT=0`. Separate
clean-context Terra/high contract and implementation reviews found no blockers.

Accepted artifact hashes and handoff are in `CURRENT.md`. Authentication uses
already verified local completion evidence, not a fresh remote-state query or
new scan of payload bytes. At runtime the reused reader validates mounted size,
header and hashed sidecars; it does not rehash entire binary payloads. Test
features remain sealed, and both existing supervised views and all pre-existing
source files remain byte-identical.
