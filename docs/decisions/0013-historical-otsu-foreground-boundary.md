# 0013: Historical Otsu Foreground Is Not Exhaustive Tissue

Status: active

## Context

The accepted VAE train/validation data and the later foreground populations use
the same historical selector. Each official WSI thumbnail is converted to HSV,
Otsu is applied to the saturation channel, and a full-resolution grid patch is
retained when its projected thumbnail rectangle exceeds the nominal 60% tissue
threshold.

Black versus white thumbnail background is not an inversion bug: both have
near-zero HSV saturation and therefore fall on the background side of the
binary threshold. The selector nevertheless has two documented limitations.

1. Saturation-only thresholding misses pale tissue. In the 152-WSI supplemental-
   mask atlas, 56,488 of 606,141 patches with at least 60% annotated area and at
   least 95% single-tissue purity were `mask` rather than `mask+otsu`. This is a
   9.32% miss rate against that annotated high-coverage reference, present in
   145 of 152 WSIs. The classwise rates are 8.68% for tumor, 17.27% for stroma
   and 4.64% for necrosis.
2. The historical implementation compares an integer-cropped thumbnail region
   with 60% of a continuous projected area. The effective cutoff therefore
   depends slightly on coordinate rounding. For WSI 59031 it ranges from 56.7%
   to 65.1%, rather than remaining exactly 60%.

The evidence comes from
`runs/kaggle/ubc_ocean_test_atlas_v3/dataset/ubc_ocean_test_atlas.csv`, whose
accepted SHA-256 is recorded in `docs/data/ubc_ocean_eval_split_audit.json`.
The audit treats a patch as the reference tissue case only when
`annotated_fraction >= 0.60` and dominant-tissue purity is at least 0.95.
Supplemental masks are incomplete, so they can measure these false negatives
but cannot establish the selector's false-positive rate.

## Decision

- Existing checkpoints, latents and evaluations remain historical accepted
  artifacts. Do not alter their membership or reinterpret them as exhaustive
  tissue coverage.
- Name the affected population **historical Otsu-selected foreground**. A
  "complete bag" means complete with respect to that frozen selector, not all
  tissue visible in the WSI.
- Tissue recognition includes eligible high-purity `mask` rows and is therefore
  not restricted to Otsu-selected coordinates. Reconstruction and WSI diagnosis
  are restricted to the historical Otsu population. The VAE development set
  used the same selector without masks, so its pale-tissue miss rate is unknown.
- Any proposal to retrain a VAE, re-extract latents, rebuild WSI bags, redefine a
  reconstruction population or transfer the pipeline to new WSIs must raise
  this foreground-selection flag before execution. It must not silently reuse
  the historical selector as a generally valid tissue detector.
- A changed selector is a new data contract. Specify it before execution,
  develop it without sealed-test labels, freeze it before test evaluation, and
  apply the same frozen coordinate manifest to both VAE branches.
- A replacement should, at minimum, address pale tissue, black and white
  backgrounds, colored artifacts and exact projected-area accounting. The
  decision may be to reproduce the historical selector, but that choice and its
  population boundary must be explicit.

This note is a rerun/retraining gate, not authorization to tune on sealed-test
evidence or to replace the accepted checkpoints.
