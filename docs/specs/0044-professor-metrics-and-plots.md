# Spec 0044: Professor-Requested Metrics And Plots

Status: implemented / locally verified
Implementation readiness: complete; final clean-context review found no P0/P1 findings
Owner/workstream: local synthesis of the completed frozen-VAE evaluation evidence
Last updated: 2026-09-06

## Purpose

Turn the immutable completed normal- and continuous-`SO(2)`-VAE artifacts into
the metrics, boxplots, dashboard, and comparison views requested in GitHub issues
#3 and #4. This is post-hoc rendering only. It must not train, tune, select a
checkpoint, access a sealed label, or launch a remote job.

The first accepted package is deliberately local under
`runs/local/professor_metrics_v1`. Paper integration, GitHub issue comments, and
Overleaf synchronization remain separate user-authorized actions.

## Non-Goals

- No model inference, optimizer update, checkpoint change, or Kaggle action.
- No metric recomputation over the full 30,000-patch VAE validation population:
  the archived validation CSVs contain aggregate rows, not the per-image values
  required for honest full-population boxplots. A future frozen-checkpoint
  inference-only evaluator may add that evidence under a separate locked contract.
- No use of the 23-WSI MIL/tissue test labels or predictions to choose images,
  styling, checkpoints, metrics, or claims.
- No paper, thesis, GitHub issue, or Overleaf mutation in this spec.
- No new scientific significance test on the fixed validation 25. They are a
  balanced qualitative panel, not an independent sealed test population.
- No claim that a PCA image, dashboard trace, or fixed-25 result proves general
  model superiority.

## Inputs And Data Contract

### Frozen final artifacts

- Normal VAE final artifact root:
  `runs/kaggle/selected_runtime_full_v4_session3/artifacts/fixed25`.
- `SO(2)` VAE final artifact root:
  `runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1/artifacts/fixed25`.
- Both final views use `boundary_060000` and the same byte-identical
  `originals.pt`, SHA-256
  `8d46560d2294b9d1ae46e9f96f2b7d55eccbcbf2757e87e6eec02aeadc54c582`.
- Normal final `reconstruction_progress.pt` SHA-256:
  `33a2fc604c05c2df79fece8ef7517185826c6b9edb58c0f9bbb938bfd1e46167`.
- `SO(2)` final `reconstruction_progress.pt` SHA-256:
  `833692eb6ee17fdcc34bac237dbc559a108f8dfd7c0f1788debd58fb36046a3c`.
- The renderer must validate the two fixed-25 manifests, selector SHA
  `ace244ecdd67aaa1ebc7d08065f1e4bfa3c0d54806d4f3b50fb38a3ae000447f`,
  identical ordered sample identities, real/promotable status, final boundary,
  tensor keys, shapes, dtypes, and finite values before writing output.
- Originals are `uint8 RGB [0,255]`; model-domain targets are
  `x / 255 * 2 - 1`. Reconstructions are archived raw normalized-domain FP16
  tensors and are converted to FP32 before metrics. Image-domain SSIM/PSNR use
  the shared `normalized_to_image_domain` clamp to `[0,1]`.

### Training-history sources

- The complete machine-readable source contract is
  `docs/data/spec0044_professor_metrics_inputs.json`, SHA-256
  `486627232fe843b537556264f1c81dd246fbd7564f589be7d9bc781e9ede96c8`.
  It pins every accepted path, successful-update-counter range, expected row
  count, AMP-skipped-attempt count, CSV hash, fixed-25 manifest hash, artifact
  manifest hash, completion-summary hash, every boundary reconstruction tensor,
  all six final rotated tensors, and the referenced PCA PNG.
- Normal history uses exactly `selected_runtime_full_v2` (counters 1–15,000),
  `selected_runtime_full_v3_session2` (15,001–45,000), and
  `selected_runtime_full_v4_session3` (45,001–60,000). The combined metrics
  directory is useful corroboration but is not sufficient because it omits the
  boundary reconstruction tensors needed for fixed-25 PSNR.
- `SO(2)` history uses exactly the seven accepted session directories pinned in
  that JSON contract: `so2_selected_runtime_full_v1_session1`, `v2_session2`,
  `v3_session3`, `v4_session4`, `v5_session5`, `session6_fresh_v1`, and
  `session7_fresh_v1_retry1`.
- The duplicate download mirror `so2_selected_runtime_full_v5_session5_remote`
  must be rejected/excluded.
- Train histories must contain exactly one committed row
  (`amp_step_skipped == 0`) per `(rank, successful_optimizer_update_count)` for
  counters 1 through 60,000. The two committed rank rows are averaged per
  counter. Skipped retry-attempt rows are distinct telemetry, not duplicate
  flushes: preserve them in a separate output and validate their exact counts.
- The normal run's accepted legacy physical-skip event is exactly counter 14,007
  on both ranks with `amp_step_skipped == 0`, nonfinite `grad_norm`, and zero
  `param_update_norm`. Require and disclose this one known exception; reject any
  additional committed physical-skip signature. Dashboard x-axes therefore say
  "recorded successful-update counter", not "physical optimizer updates".
- Validation/equivariance histories must cover boundaries 3,000 through 60,000
  in increments of 3,000, both ranks/views where applicable, with `n=25` for
  the fixed-25 metrics.
- Aggregate train curves by optimizer step across ranks, then apply a documented
  fixed-window display smoother only to the drawn trace. Preserve unsmoothed
  values in the output CSV. Aggregate validation rows by their declared sample
  counts. Average the three exact-quarter-turn fixed-25 headline equivariance
  rows at each boundary.

## Metric Contract

Run the existing shared functions from `eqvae.metrics.reconstruction` on each
of the same 25 final clean reconstruction pairs for both models:

- `mae_norm`: per-image MAE in normalized `[-1,1]` coordinates;
- `mse_norm`: per-image MSE in normalized coordinates;
- `psnr_img`: per-image PSNR after the documented image-domain projection;
- `ssim_img`: per-image SSIM after the same projection.

Write every per-image value with model, fixed selector rank, sample identity,
WSI, source label, and metric domains. Summaries use population standard
deviation, mean, median, Q1, Q3, minimum, maximum, and `n=25`. Report paired
normal-minus-`SO(2)` descriptive deltas, but no p-value or superiority claim.

The dashboard uses only existing histories:

1. smoothed train objective plus clean validation objective;
2. smoothed train L1 plus clean validation L1;
3. smoothed train `1-SSIM` loss plus clean validation `1-SSIM` loss;
4. fixed-25 clean-reconstruction PSNR at each archived boundary;
5. fixed-25 headline latent equivariance ratio averaged over 90/180/270 degrees
   on a log y-axis;
6. learning-rate schedule.

Every panel must name its population and avoid presenting fixed-25 PSNR as a
30,000-patch validation statistic. Lower/higher-is-better direction must be
visible where it is not obvious.

## Outputs And Acceptance Artifacts

All outputs are generated atomically under `runs/local/professor_metrics_v1`:

- `metrics/reconstruction_fixed25.csv`: 50 rows, one per model and fixed patch,
  with all four per-image metrics and identities;
- `metrics/reconstruction_fixed25_summary.json`: descriptive summaries and
  paired deltas;
- `metrics/training_dashboard_series.csv`: the exact unsmoothed/aggregated
  source series used by the dashboard;
- `metrics/amp_skipped_attempts.csv`: every filtered retry-attempt row, retained
  as execution telemetry and excluded from scientific train curves;
- `tables/metrics_summary.tex`: model-by-metric mean, population SD, and `n`;
- `figures/metrics_boxplots_fixed25.png`: four panels, common model colors,
  raw paired points/lines behind the box summaries, `n=25`, and explicit domains;
- `figures/training_dashboard.png`: the six-panel issue-derived dashboard;
- `figures/reconstructions_fixed25.png`: the same ordered 5x5 originals, normal
  reconstructions, and `SO(2)` reconstructions in three labeled panels;
- `figures/rotated_input_vs_latent.png`: the same predetermined fixed patch and
  0/90/180/270-degree rows for both models, with clear path labels and error
  context. This is fixed selector rank 12, sample identity
  `validation:00028360:59031:2:44544:9984`; it cannot be selected after viewing
  model outputs;
- `figures/latent_pca_reference.txt`: a manifest pointer to the already verified
  paired Spec 0038/0040 PCA artifacts rather than a misleading new PCA fit;
- `manifest.json`: schema, interpretation boundaries, exact input/output hashes,
  source paths, metric definitions, sample identities, and generation command;
- `status.json`: pass/fail acceptance summary.

PNG figures must be legible at 100% scale, use color plus line/marker/style
differences, and carry enough labels to stand alone. They are candidate source
artifacts, not automatically paper figures.

## Related Requirements And Evidence

- GitHub issue #3: shared SSIM/MAE/MSE/PSNR evaluator, mean/SD/`n`, boxplots,
  and six-panel training dashboard.
- GitHub issue #4: fixed 25 reconstructions, rotated-input versus
  transformed-latent reconstruction, all metrics/boxplots, and EQ-VAE-style
  latent views.
- The issue #3/#4 dashboard attachment was visually rechecked on 2026-09-06 and
  matches the six panels recorded in `docs/issue_image_inventory.md` and Spec
  0010. The live issue #4 text was also rechecked. Two old direct attachment
  locators currently return 404; their prior visual inspection remains recorded
  in Spec 0010 and is not silently reconstructed from memory.
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md` fixes the
  evaluation-only `rot90` interpretation.
- Specs 0010, 0038, and 0040 own the archived fixed-25/equivariance/PCA evidence.
- Specs 0041–0043 own the completed sealed downstream tests and their strict
  no-selection boundary.

## Architecture Or Workflow Contract

- Add one repo-owned evaluation module and one thin CLI. The module performs all
  validation, metric computation, source-series assembly, drawing, hashing, and
  atomic publication; the CLI only parses explicit paths and invokes it.
- Use only locked dependencies already in `pyproject.toml`/`uv.lock` (stdlib,
  NumPy, Torch, Pillow transitively present through scikit-image). Do not add a
  plotting dependency merely for this package.
- Use Pillow for deterministic raster composition and plot drawing. Centralize
  plot scales, labels, colors, fonts, and quantile definitions.
- Fail closed before publication on any hash, identity, schema, coverage,
  duplicate-session, nonfinite, shape, domain, or expected-count mismatch.
- Publish through a unique `mkdtemp` sibling directory, remove that temporary
  directory on failure, refuse any pre-existing final directory, and use a
  single no-replace final-directory rename. Concurrent invocations must yield
  at most one accepted final package and may not merge files.
- `manifest.json` hashes every scientific/table/figure/pointer output but not
  itself or `status.json`, avoiding self-reference. `status.json` records the
  final manifest SHA-256 and acceptance result; neither file is included in the
  manifest's `output_files` hash map.
- Do not read a checkpoint or source image dataset; use only the archived final
  tensors and histories enumerated above.

## Config Contract

The CLI accepts an explicit `--input-contract`, `--output-dir`, and display-only
`--smoothing-window`. Defaults point to the canonical paths in the pinned JSON
contract. The smoothing window is recorded in the manifest and cannot change
any metric table or validation point. Production execution rejects a contract
whose bytes do not match the SHA pinned above.

## Acceptance Criteria

1. All source hashes, schemas, identities, steps, views, ranks, angles, shapes,
   dtypes, and finite-value gates pass before output publication.
2. Both models use the exact same ordered 25 targets and shared metric functions.
3. The per-image CSV has exactly 50 unique `(model, selector_rank)` rows and the
   four official metrics reproduce direct shared-library recomputation.
4. Every summary/boxplot group reports `n=25`; metric domain and direction are
   explicit.
5. The dashboard covers the complete 1–60,000-step trajectories and all twenty
   validation/fixed-25 boundaries without using the duplicate `session5_remote`.
6. The PSNR dashboard panel is explicitly fixed-25; no full-population boxplot
   or significance claim is implied.
7. All four figures pass visual inspection for cropping, overlap, readable
   legends/axes, model distinction, and truthful captions.
8. Manifest output hashes revalidate exactly and a second invocation refuses to
   overwrite the accepted package.
9. Skipped AMP attempts match the pinned 38 normal and 48 `SO(2)` rows, never
   enter a scientific trace, and remain available as telemetry. The sole normal
   counter-14,007 physical-skip exception is present exactly as specified.
10. Focused tests, Ruff, formatting, BasedPyright, diff hygiene, and workspace
   preflight pass. The repo-wide gate is reported separately if still blocked by
   unrelated pre-existing findings.

## Tests And Verification Commands

```bash
.venv/bin/pytest -q tests/test_professor_metrics.py
.venv/bin/ruff check src/eqvae/evaluation/professor_metrics.py \
  src/eqvae/cli/render_professor_metrics.py tests/test_professor_metrics.py
.venv/bin/ruff format --check src/eqvae/evaluation/professor_metrics.py \
  src/eqvae/cli/render_professor_metrics.py tests/test_professor_metrics.py
.venv/bin/basedpyright src/eqvae/evaluation/professor_metrics.py \
  src/eqvae/cli/render_professor_metrics.py tests/test_professor_metrics.py
.venv/bin/python -m eqvae.cli.render_professor_metrics
git diff --check
../agent_preflight.sh
```

## Implementation Results

The accepted local package is `runs/local/professor_metrics_v1`; its manifest
SHA-256 is
`28b5d5437b213c1e5822c1553a17bcc0cbea182e625503279fa59dcb28d39a06`.
All 118 source hashes and all 10 declared output hashes revalidate. The package
contains 50 paired reconstruction rows, complete 1–60,000 dashboard histories,
all 20 validation/fixed-25 boundaries per model, and the separately preserved
38 normal plus 48 `SO(2)` AMP retry attempts.

On the predetermined fixed validation 25, normal/`SO(2)` mean (population SD)
is MAE `0.060803 (0.012743)` / `0.062416 (0.012974)`, MSE
`0.007060 (0.002769)` / `0.007474 (0.002914)`, PSNR
`27.8981 (1.8394)` / `27.6451 (1.8260)` dB, and SSIM
`0.730803 (0.063662)` / `0.717021 (0.063675)`. These are descriptive
fixed-panel results, not full-validation or sealed-test estimates. The
predetermined rotated-path figure shows much smaller transformed-latent versus
rotated-input MAE disagreement for the `SO(2)` model at 90/180/270 degrees,
while the archived all-25 headline equivariance ratio itself does not favor
`SO(2)`; the package therefore preserves both views without a superiority
claim.

Seven focused tests, Ruff, formatting, BasedPyright, output-hash validation,
visual inspection of all four PNGs, and `git diff --check` pass. The final
independent adversarial review found no P0/P1 findings. The repo-wide Python
gate remains blocked by 23 unrelated pre-existing lint findings in the tissue
fastpath probe and packaged full-compile kernel. A non-blocking code-quality
debt remains: the large dynamic JSON/Pillow renderer still uses broad
file-level BasedPyright suppressions and should be narrowed if this module is
later extended.

## Implementation Blockers

- None for the locked local renderer. Full-population validation boxplots still
  require a separate frozen-checkpoint inference contract and source data.

## Known Risks

- Fixed-25 distributions are useful for the advisor-requested qualitative panel
  but are not a substitute for a full validation or sealed test distribution.
- FP16 archive quantization introduces small reconstruction-metric differences
  from live FP32 inference; the manifest and captions must disclose this.
- Skipped AMP attempts share successful counters with later committed rows;
  naive counter deduplication can select the wrong observation. Session mirrors
  can also double-count valid rows.
- A visually smooth PCA color map is basis/sign/scale dependent and cannot rank
  the models by itself.
- Dense train curves can conceal transient behavior if aggressively smoothed;
  preserve raw aggregates and fix/document the display window.

## Adversarial Checks

- Swap one original identity, source root, manifest, or reconstruction tensor
  between models and require a pre-publication failure.
- Add the duplicate `session5_remote`, remove a step/boundary/rank/view, alter an
  input hash, misclassify a skipped attempt, or introduce a second physical-skip
  signature and require a coverage failure.
- Race two renderers against the same absent destination; require one complete
  package and one refusal, with no merged or orphaned temporary output.
- Mutate normalized/image-domain handling and ensure the known tensor vector
  catches the error.
- Verify that boxplot labels, table headings, and JSON keys cannot confuse
  normalized-coordinate MAE/MSE with image-domain PSNR/SSIM.
- Verify that generated prose never calls fixed-25 validation a sealed test or a
  statistically significant result.

## Open Questions

- A full 30,000-patch frozen validation metric distribution remains a separate
  possible inference-only workstream because per-image values were not archived.
- Moving accepted candidate figures/tables into `paper/sipaim2026`, compiling the
  PDF, and posting Spanish issue updates require explicit follow-up scope.

## Related Files

- `GOAL.md`
- `CURRENT.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`
- `docs/specs/0010-fixed25-equivariance-artifact-protocol.md`
- `docs/specs/0038-frozen-vae-rotation-orbit-visualization.md`
- `docs/specs/0040-spatial-latent-pca-coherence-visualization.md`
