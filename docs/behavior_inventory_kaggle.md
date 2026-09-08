# Kaggle Data And Behavior Contract

Status: current
Last updated: 2026-09-08

## Purpose

Record stable source, format and split behavior needed by current code. Remote
execution rules live in `docs/kaggle_cli_workflow.md`; exact accepted artifact
identities live in `CURRENT.md` and the relevant specs.

## Canonical Sources

| Source | Canonical locator | Current role |
| --- | --- | --- |
| UBC-OCEAN competition | `UBC-OCEAN` | Official WSI images, labels and metadata |
| Supplemental masks | `sohier/ubc-ovarian-cancer-competition-supplemental-masks` | Non-exhaustive tumor/stroma/necrosis annotations for the 152-WSI held-out cohort |
| Pre-shuffled patches | `maximusshtefan/patches-pre-shuffled-ubc-ocean` | Unsupervised train/validation source |
| Raw atlas | `maximusshtefan/raw-atlas-ubc-ocean` | Coordinate provenance |
| Train/validation atlas | `maximusshtefan/train-val-atlas-ubc-ocean` | Development split provenance |

Never replace a canonical owner with the authenticated Kaggle username. The
authenticated user owns only newly created resources. Pin positive versions and
receipts for every immutable private input or output.

## Development Patch Source

The pre-shuffled dataset contains:

```text
dataset/ubc_train_shuffled.bin
dataset/ubc_train_shuffled.csv
dataset/ubc_ocean_valid.bin
dataset/ubc_ocean_valid.csv
```

| Split | WSIs | Patches |
| --- | ---: | ---: |
| Train | 322 | 300,000 |
| Validation | 39 | 30,000 |

The binary format has a 64-byte header followed by CHW RGB `uint8` payloads of
shape `3x256x256`. The train and validation WSIs are disjoint and have no
overlap with the 152 supplemental-mask WSIs. This dataset has no sealed-test
shard; sealed evaluation uses the separate coordinate and latent contracts.

Patch-label mapping:

| Numeric label | Diagnosis |
| ---: | --- |
| 0 | CC |
| 1 | EC |
| 2 | HGSC |
| 3 | LGSC |
| 4 | MC |

## Held-Out WSI Contract

The 152 masked non-TMA WSIs form the fixed evaluation cohort. Exact membership
is `docs/data/ubc_ocean_masked_holdout_ids.csv`.

- Shared WSI split: 106 train, 23 validation, 23 sealed test.
- Full foreground population: 1,750,221 Otsu-selected coordinates.
- Tissue population: 666,807 high-purity annotated coordinates.
- Full reconstruction-test population: 67,138 cancer-task patches from the 23
  sealed-test WSIs.
- TMA images are excluded.

Mask colors are red=tumor, green=stroma and blue=necrosis. Black means
unannotated, not normal or negative. The full-foreground rule is image-derived
Otsu tissue selection; masks add tissue-class information but do not define
exhaustive WSI coverage.

Specs 0017–0019 own the exact coordinates, split constraints, work manifests
and hashes. Specs 0020–0025 own immutable latent storage and logical views.

## Latent Storage

- Each physical record is posterior `mu` in FP32 with shape `[16,32,32]`.
- Normal and `SO(2)` stores are separate and immutable.
- Task/split views reference physical parts by logical row ranges; they do not
  duplicate binary payloads.
- Extraction streams official WSI pixels directly into latent stores; it does
  not create a raw RGB sealed-test shard.
- Every producer and consumer validates row identity, order, shape, byte size
  and cryptographic receipts before use.

## Training Input Semantics

- Images are normalized from RGB `uint8` into `[-1,1]`.
- Stain/noise corruption creates the denoising input; the clean normalized patch
  remains the reconstruction target.
- The normal and `SO(2)` models receive the same examples, ordering contract,
  objective and validation schedule.
- Runtime artifacts record committed optimizer updates separately from physical
  attempts and AMP skips.

The executable implementation is under `src/eqvae`; repo-managed script kernels
are under `kaggle/kernels`. Notebook exports and `reference/` are not executable
sources of truth.

## Sealed-Test Boundary

MIL, tissue and reconstruction tests use label-blind remote prediction followed
by receipt-bound local scoring. Their one-shot authorities are consumed.
Predictions, scores and bootstrap outputs may be reported but may not alter
models, checkpoints, thresholds, hyperparameters or retry decisions.
