# Spec 0045: Frozen-VAE Test Reconstruction Evaluation

Status: protocol accepted and local prelaunch implementation complete
Implementation readiness: label-free private input and compact dual-T4 kernel validate locally; no Kaggle write is authorized
Owner/workstream: sealed-test reconstruction metrics for the two frozen VAEs
Last updated: 2026-09-06

## Purpose

Evaluate the final normal and continuous-`SO(2)` VAEs on the complete, already
frozen Spec 0019 cancer/autoencoder test population rather than treating the 25
fixed validation patches as population evidence. The evaluation measures clean,
deterministic mean-posterior reconstruction only and produces the full-test
metrics and plots requested by the professor.

The fixed 25 remain the qualitative validation panel. This spec adds a distinct
quantitative test result over 67,138 patches from 23 sealed-test WSIs.
The main question is reconstruction fidelity over those patches. Diagnosis is
used only afterward as a secondary diagnostic grouping.

## Non-Goals

- No training, checkpoint selection, hyperparameter tuning, corruption, test-time
  augmentation, stochastic posterior sample, or output `tanh`.
- No KL, `logvar`, likelihood, FID, or sampled-posterior claim. Spec 0021 did not
  retain `logvar`; this work is therefore a full-test **reconstruction**
  evaluation, not a complete generative-model evaluation.
- No independent-patch confidence interval or claim that 67,138 correlated
  patches are 67,138 independent experimental units.
- No model winner selected from four uncorrected endpoints. MAE is the one
  confirmatory endpoint; MSE, PSNR, and SSIM are prespecified secondary metrics.
- No paper, thesis, GitHub issue, or Overleaf mutation.
- No remote dataset publication or Kaggle launch without fresh explicit user
  permission and the existing write confirmations.

## Inputs And Data Contract

### Frozen population

- Authoritative manifest:
  `runs/local/ubc_ocean_eval_consumption/cancer_test.csv`.
- Exact SHA-256:
  `1d0e4059f469d350ff3960cc10208221548f6afdfc1788e40e6d5da7829806cc`.
- Exactly 67,138 centered-systematic, at-most-3,000-per-WSI, image-derived
  Otsu patches from the 23 Spec 0018 test WSIs. Mask-only coordinates are absent.
- Patch/WSI support is fixed before this evaluation:

| Diagnosis | Patches | WSIs |
|---|---:|---:|
| CC | 15,000 | 5 |
| EC | 16,138 | 6 |
| HGSC | 24,000 | 8 |
| LGSC | 6,000 | 2 |
| MC | 6,000 | 2 |

Twenty-two WSIs contribute 3,000 patches; EC WSI 50048 contributes 1,138.
The population and four reconstruction metrics predate test release, but the
exact aggregation and bootstrap contract in this spec is being fixed after the
separate downstream test labels were opened. That timing must be disclosed.

### Frozen latent and decoder inputs

- Use the existing Spec 0021 FP32 posterior means from the five exact v1 kernel
  outputs `maximusshtefan/eqvae-ubc-ocean-latent-run-01` through `-05`.
  Do not republish or locally copy their roughly 73 GiB paired payload.
- Global latent audit:
  `runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/spec0021_latent_store_global_audit.json`,
  SHA-256
  `25dca0a379da88c41a99bf738908cbd22b89eed94b0193bc972832dde6e9889a`.
- Cancer-test location source:
  `runs/kaggle/ubc_ocean_latent_store_finalizer/dataset/views/cancer_test_locations.csv`,
  SHA-256
  `3599baeb4b70d0414e3f7fa9bf8a359c91b2433613dde186b58f6a711f95e66b`.
- A new local input builder must deterministically redact that location table to
  `run_number,file_index,atlas_row_index,wsi_id,x,y,split`, prove row-for-row
  identity against the authoritative cancer-test manifest, and package no
  diagnosis or tissue field.
- Normal checkpoint source:
  `runs/kaggle/selected_runtime_full_v4_session3/checkpoints/step_060000.pt`,
  SHA-256
  `f733304e9178e468546113642bdf01e11348570b340c366cf148973083cb9075`.
- `SO(2)` checkpoint source:
  `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/checkpoints/step_060000.pt`,
  SHA-256
  `041e0cd7483cb8642bb72eb1b63c3a36774bf9cadd0b659c9d1db6a813c8f4c7`.
- The new private input contains only the redacted locations, two strictly
  derived state-dict-only weight files, and its byte manifest. It must contain
  no test diagnosis/tissue labels. The source checkpoint hash, derived state
  identity, and derived file hash are all recorded. These payload files are
  deliberately flat at the dataset root so the Kaggle CLI uploads every file
  rather than silently skipping nested directories.
- Official UBC-OCEAN `train_images/{wsi_id}.png` supplies the target pixels.
  Each of the 23 source PNG byte lengths and SHA-256 values is pinned from the
  accepted Spec 0021 pair audits and checked during the run.
- Kaggle's official UBC-OCEAN mount also contains `train.csv` with diagnosis
  labels. Strict environment-level label isolation is therefore impossible
  while using the official images. The frozen embedded evaluator resolves only
  `train_images`, never opens any competition CSV, emits no diagnosis/tissue
  field, and is byte-authenticated before the local diagnosis join. This is an
  auditable code-level non-use boundary, not a claim that labels are absent from
  the whole runtime mount.

## Reconstruction And Metric Contract

For each exact location and each model:

1. read the stored FP32 posterior mean `mu`;
2. set the frozen model to evaluation mode and disable gradients;
3. compute the raw decoder output `D(mu)` in FP32;
4. normalize the target exactly as `uint8 / 127.5 - 1`;
5. compute per-image metrics through `eqvae.metrics.reconstruction`:
   - `mae_norm` and `mse_norm` on raw normalized `[-1,1]` values;
   - `psnr_img` and `ssim_img` after the shared clamp/project to `[0,1]`.

PSNR is calculated per image and then summarized; it is never reconstructed
from an aggregate MSE. If any per-image PSNR is positive infinity, preserve and
report its exact count, never cap it, and omit PSNR bootstrap intervals.

## Estimands And Diagnostic Breakdown

The primary reconstruction result for each model is pooled patch-level MAE:

`mean of the 67,138 per-patch MAE values`.

The confirmatory comparison is the paired normal-minus-`SO(2)` difference in
that same pooled patch mean. Lower MAE is better, so a negative difference
favors the normal VAE and a positive difference favors the `SO(2)` VAE. MSE,
per-image PSNR, and SSIM are prespecified secondary reconstruction metrics.

For every metric and model, report the patch-level population mean, population
SD, median, quartiles, min, max, and infinity count where applicable. Also
report the unweighted mean of the 23 WSI means as a robustness view; it does not
replace the primary patch-level result.

Diagnosis is a secondary diagnostic breakdown only. For CC, EC, HGSC, LGSC,
and MC separately, report the patch distribution, pooled patch mean, WSI-macro
mean, patch count, and WSI count. This asks whether reconstruction fidelity
changes with diagnosis; diagnosis prevalence does not rebalance or define the
overall primary score.

The 95% interval for the paired pooled-patch MAE difference uses 10,000 paired,
unstratified WSI-cluster bootstrap draws with frozen seed 4501. Each draw
resamples 23 WSIs with replacement, uses the same WSI multiplicities for both
models and all metrics, and reconstructs the pooled-patch mean by weighting
each sampled WSI mean by its patch count. It never resamples patches
independently. Secondary intervals are exploratory and do not define an
all-metric winner. Bootstrap uncertainty covers resampling of these WSIs only,
not training seeds or alternative patch selectors.

LGSC and MC each contain only two WSIs, so diagnosis-specific comparisons are
descriptive and may be unstable. Per-patch SD is dispersion, not inferential
uncertainty.

## Remote Workflow Contract

1. Build and locally validate a private, diagnosis/tissue-free input dataset.
2. Freeze the spec, shared metric/data contract, runtime evaluator, scorer, scorer vector,
   reporter, redacted locations, input
   contract, kernel bytes, metadata bytes, exact five v1 latent sources, and
   official WSI identities before launch.
3. Launch one private, dual-T4, inference-only kernel. The run authenticates all
   ten mounted latent binaries against their accepted full-file hashes, performs zero
   optimizer updates, and its frozen code never reads a diagnosis source. It installs
   only the already exercised PyTorch `2.14.0+cu130` build rather than floating to a
   later release; unused `torchvision` and `torchaudio` are not installed.
4. Read each exact target patch once and decode the two stored posterior means
   concurrently on the two GPUs. Padding may preserve a fixed decoder batch
   shape but padded rows never enter output.
5. Publish only after all 67,138 paired rows, all 23 WSI hashes, all finite
   non-PSNR metrics, exact row order, exact source identities, and zero-update
   checks pass. `status.json` is written last.
6. A byte-identical retry is allowed only for an infrastructure failure before
   any metric artifact is exposed. After a complete metric result is exposed,
   the one-shot evaluation authority is consumed.
7. Authenticate the downloaded output and write an exclusive pre-score claim
   before joining the local diagnosis oracle or computing any aggregate result.

## Outputs And Acceptance Artifacts

Remote output:

- `per_patch_metrics.csv.gz`: one paired row per exact patch with location
  identity and both models' MAE/MSE/PSNR/SSIM, but no diagnosis/tissue label;
- `wsi_evidence.json`: the 23 checked PNG identities and row transcripts;
- `runtime.json`: versions, dual-T4 identity, FP32 policy, timing, and zero
  optimizer updates;
- `run_contract.json`: every frozen input/code/source hash;
- `manifest.json` and status-last `status.json`.

Receipt-authenticated local package:

- exclusive `pre_score_claim.json` written before label join;
- `metrics/per_patch_metrics_joined.csv.gz`;
- `metrics/per_wsi_metrics.csv` (23 WSIs x two models);
- `metrics/per_diagnosis_summary.csv`;
- `metrics/overall_summary.json` with the primary patch distribution and
  WSI-macro robustness view;
- `metrics/paired_bootstrap.csv.gz` and interval summary;
- `tables/vae_test_metrics.tex`;
- `figures/patch_metric_boxplots.png` with the correlated-patch/descriptive note;
- `figures/patch_mae_by_diagnosis.png` with diagnosis marked as exploratory;
- `figures/paired_wsi_mae.png` with one paired line per WSI;
- completion `manifest.json` and status-last `status.json`.

## Acceptance Criteria

1. Population hash, 67,138 row count, exact order, 23 WSI count, class support,
   and 22x3,000 plus 1x1,138 patch support all match this spec.
2. Mounted evaluation input contains no diagnosis/tissue/target/truth field or
   file and matches an immutable private dataset receipt.
3. Exact v1 latent sources, ten binary/sidecar records, global audit, location
   source, redacted location file, model source checkpoints, derived state
   identities, and 23 WSI PNG identities are all bound before launch.
4. Both branches use the identical ordered rows, stored FP32 `mu`, clean FP32
   `D(mu)`, and shared metric functions; optimizer updates equal zero.
5. Remote publication is atomic and contains exactly 67,138 paired metric rows
   plus the declared provenance/status files.
6. The pre-score claim exists before the local scorer reads the diagnosis
   oracle; remote files contain no test diagnosis or tissue label, current
   spec/evaluator/scorer/vector/reporter hashes equal their prelaunch contract,
   and any post-freeze mutation fails closed.
7. Primary pooled-patch MAE, its paired WSI-cluster interval, and the separate
   per-diagnosis patch summaries reproduce the frozen scorer vector. No
   independent-patch interval exists.
8. All summaries disclose aggregation unit, metric domain, direction, WSI and
   patch support, sparse LGSC/MC strata, and lack of training-seed uncertainty.
9. Figures and TeX table are legible and report full-test rather than fixed-25
   scope.
10. Focused tests, touched-file Ruff/format/BasedPyright, kernel validation,
    `bash -n scripts/kaggle_kernel.sh`, `git diff --check`, repo preflight, and
    the applicable Python quality gate pass or have a precisely scoped
    pre-existing blocker.

## Tests And Verification Commands

The implementation must provide focused tests for redaction, exact identity
alignment, latent row resolution, metric domains, infinity handling, pooled
patch aggregation, secondary diagnosis summaries, paired WSI-cluster bootstrap
determinism, remote output authentication, status-last publication, and
mutation failures.

```bash
.venv/bin/pytest -q tests/test_spec0045_vae_test_evaluation.py \
  tests/test_spec0045_vae_test_scoring.py
.venv/bin/ruff check <touched Python files>
.venv/bin/ruff format --check <touched Python files>
.venv/bin/basedpyright <touched Python files>
bash -n scripts/kaggle_kernel.sh
git diff --check
./scripts/agent_preflight.sh
```

## Implementation Blockers

- No local blocker. The byte-frozen local input is
  `runs/local/vae_test_evaluation_input`; the uploadable kernel is
  `kaggle/kernels/vae_test_reconstruction`. Remote input publication and the
  evaluation launch each require fresh explicit user authorization and their
  write guards.

## Known Risks

- The run reads roughly 8.8 GiB of selected FP32 latents plus target pixels from
  very large WSI PNGs; source-mount throughput may dominate GPU decoding.
- Existing latent payloads are large remote kernel outputs. A launch claim must
  pin their exact version 1 sources; a mutable/latest source is insufficient.
- The official image mount includes `train.csv`; label isolation is enforced by
  authenticated evaluator non-use and output schemas, not by mount-wide absence.
- A green mechanics gate cannot repair an ill-posed inferential unit. The point
  estimate is patch-level; uncertainty therefore resamples WSI clusters rather
  than pretending patches from one slide are independent.
- This work measures reconstruction fidelity, not necessarily representation
  usefulness. Specs 0041 and 0043 remain the held-out downstream evaluations.

## Adversarial Checks

- Replace cancer-test locations with tissue-test or full-foreground rows and
  require failure before inference.
- Leak one diagnosis field into the mounted input or remote output and require
  failure.
- Swap one run/file index, WSI coordinate, latent source version, state dict,
  PNG byte, or metric domain and require failure.
- Bootstrap patches instead of WSIs, replace the primary pooled-patch mean with
  diagnosis balancing, or omit diagnosis-specific support and require a
  scorer-vector mismatch.
- Compute PSNR from mean MSE, cap infinity, or claim four-endpoint superiority
  and require review rejection.
- Publish one model without the other or omit a final row/status and require the
  whole remote package to remain unaccepted.

## Related Files

- `docs/specs/0018-shared-masked-wsi-evaluation-split.md`
- `docs/specs/0019-task-consumption-manifests-and-kaggle-work-plan.md`
- `docs/specs/0020-fp32-latent-shard-format-and-io.md`
- `docs/specs/0021-dual-model-wsi-latent-inference.md`
- `docs/specs/0044-professor-metrics-and-plots.md`
- `src/eqvae/metrics/reconstruction.py`
- `CURRENT.md`
- `GOAL.md`
