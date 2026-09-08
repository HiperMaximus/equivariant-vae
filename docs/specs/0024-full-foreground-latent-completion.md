# Spec 0024: Full-Foreground Latent Completion

Status: implemented / locally verified; publication and extraction pending
Owner/workstream: complete frozen paired embeddings for all 152 masked WSIs
Last updated: 2026-08-30

## Scope

The user requests completing coverage across the fixed 106/23/23
train/validation/test split after the full-WSI transformer capacity pass.
Interpret full coverage consistently with that probe: all image-derived Otsu
coordinates from Spec 0018, including `mask+otsu` but excluding mask-only
coordinates. Do not reinterpret this as all 1,822,340 annotated atlas rows.
This is frozen feature extraction, not classifier training or test release.
Remote publication and each launch still require explicit approval.

## Inputs And Inventory

Use the existing authenticated manifests, not binary reads, to subtract stored
identities `(atlas_row_index,wsi_id,x,y)` from the fixed candidate set:

| Source | Reusable foreground rows per model |
| --- | ---: |
| Five base parts | 576,375 |
| Part 11 | 74,033 |
| Completed WSI45630 supplement | 22,649 |
| Remaining to encode | 1,077,164 |
| Full target | 1,750,221 |

Metadata-only verification on 2026-08-30 authenticated all four manifests,
checked strict numeric `wsi_id,y,x` order, unique identities/coordinates and
disjoint stored sources. The remaining set covers 145 WSIs; the largest
missing WSI has 20,924 rows. WSI45630 has zero missing rows. The largest
foreground bag across all 152 WSIs is 32,595, equal to the tested bag size.
The base contains
23,023 additional non-Otsu rows: retain them unchanged but exclude them from
MIL foreground bags. Membership is frozen before consulting storage reuse.

Candidate/base/part-11 paths and hashes are in Spec 0022. The completed
WSI45630 manifest is
`runs/local/wsi45630_completion/bundle/inference/cancer_topup_manifest.csv`,
SHA-256 `df4d5281c585bd39826a6d528eba2b8b1ddd15444e173c0c59cf3b6750202adc`.
Its verified remote pair audit and the capacity evidence are linked from
`CURRENT.md`. Authenticate completion evidence before assigning catalog parts.
The WSI45630 pair audit SHA-256 is
`267a8b8128e1a637352223ee3e5939edd9c47b29e7e8617fdbedcb1aae1ea20b`;
its worker contract is
`7b060e303a3ddc0559e7d616eaed20a983ff5287492a1d30280494f474a9481a`.
The existing top-up and WSI45630 logs respectively hash to
`88432e7897ca8675a66a3de147797ae2ff8acc68d31e8adf0038cc2361e10ccc`
and `967b6955baa432d909532543fb8b5ade2972f2b5b33b43a68fa8d2e09074af57`.
All 152 original PNG identities agree across the seven completed producers,
including 66 cross-producer WSI comparisons; no PNG payload was read locally.

## Eight-Run Execution Contract

- Reuse the existing label-free top-up worker, frozen step-60000 checkpoints,
  clean RGB normalization, FP32 posterior `mu[16,32,32]`, batch 8 and one VAE
  per T4. Read every missing RGB patch once for both models. No new inference
  tuning, raw RGB outputs, predictions, pooled vectors or latent format.
- Partition only missing rows into contiguous numeric WSI ranges, preserving
  whole WSIs and `wsi_id,y,x` inside every part. Zero-missing WSIs need no job.
  Reuse the established approximately balanced whole-WSI partition rule.
- New paired payload is 141,186,039,808 bytes (131.49 GiB), plus 1,024 header
  bytes across eight pairs. Every pair plus 10 MB metadata reserve must fit
  the 20,000,000,000-byte saved-output cap (at most 152,511 rows per job).
  Apply the existing nearest-remaining-mean greedy partition to ascending
  nonempty WSI groups: stop before the next WSI if the current row count is
  at least as close to the remaining-row/remaining-part target. Leave at least
  one WSI for each remaining part. The exact result is:

  | Run | Numeric WSI range | WSIs | New rows/model | Paired bytes incl. headers | Largest missing WSI | Inference allowance (s) | Total incl. reserve (s) |
  | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 01 | 66–8279 | 21 | 140,268 | 18,385,207,424 | 13,973 | 23,477 | 27,077 |
  | 02 | 8531–14424 | 16 | 137,761 | 18,056,609,920 | 18,987 | 23,543 | 27,143 |
  | 03 | 14542–20316 | 21 | 135,378 | 17,744,265,344 | 15,982 | 22,743 | 26,343 |
  | 04 | 21432–32432 | 23 | 131,266 | 17,205,297,280 | 13,420 | 22,125 | 25,725 |
  | 05 | 33708–39146 | 17 | 132,584 | 17,378,050,176 | 15,731 | 22,323 | 25,923 |
  | 06 | 39172–47105 | 16 | 136,829 | 17,934,450,816 | 14,665 | 22,960 | 26,560 |
  | 07 | 47960–56947 | 16 | 135,184 | 17,718,837,376 | 20,924 | 23,447 | 27,047 |
  | 08 | 57162–65533 | 15 | 127,894 | 16,763,322,496 | 16,587 | 21,701 | 25,301 |

- Timing is a conservative planning estimate, not a new benchmark or guarantee.
  Authenticate the five original pair audits/logs through the old top-up
  contract's `source_runs`, plus completed top-up and WSI45630 pair audits/logs.
  Use the last log timestamp, including bootstrap and finalization, not pure
  encoder time: base sessions 10,483.1285 / 9,573.1193 / 9,921.5744 /
  10,209.5477 / 10,207.3215 seconds; top-up 11,119.3598; WSI45630
  2,408.2699 (its pair was printed earlier at 2,402.0472).
  Let `r=max(elapsed/rows)` over all seven, `d=max(elapsed/WSIs)` over the six
  multi-WSI runs, and `s=ceil(single-WSI elapsed)`. Thus
  `r=0.15019464063778315`, `d=352.0533705117931`, `s=2409`.
  For each job let `B=max(ceil(largest_missing_WSI*r),ceil(d),s)` and
  `P=max(ceil(job_rows*r),ceil(job_WSIs*d))`. Its inference allowance is `P+B`,
  intentionally adding the largest-WSI admission margin rather than treating
  a per-WSI mean as a bound. Require `P+B <= 25200`, then add the unchanged
  3600-second reserve. Worker key `worst_observed_seconds_per_wsi` receives
  `B` (2409/2852/2409/2409/2409/2409/3143/2492); the legacy method identifier
  remains for unchanged worker compatibility, with the extended formula explicit
  in the new plan. The single-WSI session contributes to `r` and `s`, not a
  fictitious fixed 2409-second cost for every tiny WSI. The wrapper has no
  publishable resume: deadline refusal or failure loses that unfinished job;
  do not retry automatically or enlarge the locked job.
- Each run binds its own immutable manifest/config and emits a distinct pair,
  sidecars and pair-audit-last completion. Do not overwrite an earlier producer
  or regenerate any base,
  part-11 or WSI45630 row. Repeated filenames across producers require
  source-qualified catalog entries and authenticated sidecar resolution.
- Stage one shared private immutable input dataset
  `maximusshtefan/eqvae-full-foreground-inputs`, intended version 1, with exactly
  one copy of both checkpoints, the complete current `src/eqvae` Python snapshot,
  eight label-free manifests and worker contracts, and one hash/allow-list input
  contract. Preserve package initializer import closure by using the established
  full source snapshot; do not speculate about a six-file model subset.
  Generate eight thin launchers at
  `runs/local/full_foreground_completion/kernels/run_XX`, targeting distinct
  `maximusshtefan/eqvae-full-foreground-XX` producers. Each binds its run,
  manifest, worker contract, and immutable shared input contract; source/checkpoint
  contents are not embedded. The launcher assembles a temporary four-file
  inference root using symlinks to only its own manifest/contract and the shared
  read-only checkpoints, then calls unchanged `run_topup`. Source stays a sibling
  and the temporary root is removed. All launchers are below 1 MB and attach
  only the shared input dataset and official UBC-OCEAN images. Publication and
  byte-verification use the established ZIP envelope and guarded script actions;
  launch refuses an absent/mismatched private-v1 receipt. No test logical CSV,
  labels, masks, latent binaries, or old producer mount enters this bundle.
- Frozen label-free inference may encode test coordinates, as in Specs
  0021-0022. It may not read labels, compute evaluation metrics or inform any
  architecture, LR, model-selection or stopping decision. Test feature access
  remains sealed for classifier/evaluation consumers.

## Logical Integration And Ordering

After every pair is verified, produce one aggregate audit proving disjoint
stored-source identities and exact coverage of all 1,750,221 target rows.
Compact verification must bind each new producer to its planned manifest and
worker contract, validate both sidecars/counts/bytes/first-last identities and
complete WSI order, and compare `png_sha256,png_bytes` for each WSI against
its authenticated old source evidence. Reused filenames or legacy run number
`1` never identify a physical part by themselves.
Build distinct full-coverage MIL location/bag manifests and an extended catalog;
preserve the existing quarter-coverage bags/catalog unchanged. Preserve WSI
splits and identical normal/SO(2) locations. Train/validation/test remain
logical views, not binary copies. Development inputs expose train/validation
only; test mappings remain release-only. A full-coverage learning protocol and
its launch require separate specification/authorization.
Existing tissue subsets and their sealed calibration remain unchanged. Do not
silently change the separate primary VAE evaluation sample/protocol.

Ordering is sequential within each physical part, not a newly concatenated
global file. Reuse `SupervisedLatentStore.read_rows`: group reads by part and
increasing file offset, then restore requested `wsi_id,y,x` bag order. Old
parts may have gaps; no repacking or promise of one contiguous read. Classifier
WSI traversal remains governed by its matched training-order contract.

## Readiness And Verification

Implemented `src/eqvae/cli/build_ubc_full_foreground_completion.py`, the thin
`kaggle/kernels/full_foreground_completion/run_template.py`, focused tests and
guarded `scripts/kaggle_kernel.sh` actions. The local eight-run metadata build
passes; the shared source imports the unchanged worker/model registry and all
28 imported `eqvae` modules entirely from its staged snapshot. The narrow
metadata-only completion verifier writes an aggregate audit only after all eight
pairs verify; it does not implement classifier integration or activate logical
views. Full-coverage logical views/catalog remain an explicit post-extraction gate.
Independent clean-context Terra/high review rederived all eight parts and
timing values from the metadata and seven completed logs, checked compatibility
with the unchanged worker and shared symlink input, and found no concrete
contract blocker. A separate clean-context Terra/high code review found no
blocking issue in subtraction, partitioning, package/producer/receipt bindings,
unchanged-worker compatibility, guards or compact completion verification.
Five focused tests cover all-stored-row subtraction and identity drift,
whole-WSI numeric order/offsets, largest-WSI/output bounds, cross-run and label
rejection with shared checkpoint isolation, and foreign sidecar/PNG rejection.
The repository Python quality gate passed with
`PYTEST_ADDOPTS='-q -k full_foreground'`: five tests, 988 deselected,
whole-repository Ruff and BasedPyright clean. Repo preflight and diff whitespace
checks passed. No full suite, latent payload read or remote action was performed.

The final local package lives at `runs/local/full_foreground_completion`;
commands are in the frozen-posterior workflow section of
`docs/kaggle_cli_workflow.md`. Next action is separately authorized private input
publication, byte-verification of version 1, then explicitly authorized individual
launches. No receipt or new remote pair exists yet. The package binds this spec
and source snapshot; do not modify or rebuild its published bytes in place.
Keep operational progress in `CURRENT.md` until verification permits changing
the contract. Download only compact
remote evidence, never `.bin` payloads. No paper/thesis/GitHub updates, learning
campaign, generic orchestration framework or automatic retry campaign.
