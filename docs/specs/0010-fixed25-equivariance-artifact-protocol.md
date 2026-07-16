# Spec 0010: Fixed-25 Embedding-Equivariance Evaluation Protocol

Status: implemented (on working tree, uncommitted; 2026-07-01)
Implementation readiness: implemented and locally verified (gate green, 6-lens
adversarial review integrated); the REAL fixed-25 selector is generated and committed
(FU-041 DONE, 2026-07-02), so a real full run can now produce promotable fixed-25 output
Owner/workstream: comparable non-equivariant VAE baseline, qualitative and
embedding-equivariance evaluation artifacts (FU-040, token
`fixed25_equivariance_artifact_protocol`)
Last updated: 2026-07-02

## Framing: what this is (and is not)

This spec defines an **evaluation / inspection protocol**, decoupled from training.
See `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`.

- The research question is architectural: does replacing ordinary convolutions with
  continuous `SO(2)`-steerable convolutions produce a **better embedding space**? We
  compare two otherwise-matched autoencoders — the non-equivariant, non-steerable
  baseline (built first) and the future `SO(2)`-steerable model — by **inspecting
  their embedding spaces**.
- `SO(2)` steerability is a property of the equivariant model's **convolutions**,
  not of this evaluation and not an angle set. This protocol does not implement or
  train it.
- We probe the embedding space with **exact `torch.rot90` at `{0, 90, 180, 270}`
  degrees**. Following EQ-VAE (arXiv:2502.09509), the rotation-equivariance error of
  the embedding is a **proxy for how smooth / structured the embedding is**. `rot90`
  is a discrete, exact spatial permutation (no interpolation artifacts), which makes
  it a clean proxy; this is the paper's own choice ("multiples of 90 degrees to
  avoid corner artifacts", §3.3) and the FSQ reference's
  (`kaggle/train_runs:1056-1068`). Using `rot90` for evaluation does NOT conflict
  with the continuous-`SO(2)` architecture goal
  (`docs/decisions/0001-continuous-so2-scope.md`) — they are different layers.
- The protocol is **not part of training**: no loss term, no augmentation, no effect
  on the objective. It runs at evaluation points (every half-epoch boundary, and
  post-hoc on a checkpoint).
- It is **comparative**. The identical protocol runs on both models over the **same
  frozen 25 images**; the deliverable is the comparison. The baseline having higher
  equivariance error is the expected contrast, not a defect.

### Settled implementation decisions (user, 2026-07-01)

- Rotation: exact `torch.rot90` at `{0, 90, 180, 270}` degrees (decision 0009).
- Cadence: **full save at every half-epoch boundary**, FSQ-style — reconstructions,
  rotated reconstructions, latent arrays, PCA, and equivariance CSV rows are all
  written each boundary (not final-only), so the embedding evolution is inspectable
  frame-by-frame (`kaggle/train_runs:1027-1090`).
- Error/difference maps: **full-frame (unmasked) only**. `rot90` introduces no
  invalid region and the UBC patches carry no per-patch tissue mask, so there is no
  masked variant (this also matches the FSQ reference, which used no mask). The
  original "boundary-masked" requirement was written under a continuous-angle
  assumption that no longer applies.
- PCA visualization: the EQ-VAE method AND a first-3-channels fallback (below).
- The trivial `artifacts/reconstruction_samples.pt` single-patch dump is retired
  from the full run (superseded by the fixed-25 originals + reconstruction progress);
  the full-run verifier is updated to require the fixed-25 artifacts instead. The
  debug/tiny gate keeps its trivial dump unchanged (bounded proof, not paper output).
- Scale-equivariance and LPIPS are out of scope.
- Config home: a shared `fixed25_equivariance` block in
  `non_eq_vae_model_base.json`, enabled by the full-run config, so the baseline and
  the future `SO(2)` model use byte-identical evaluation config.
- Placement: one shared evaluator, called in-run at each boundary (dashboard curve)
  and exposed as a standalone entry point re-runnable on any checkpoint and on the
  `SO(2)` model.
- Spec org: standalone Spec 0010 extending the Spec 0009 runner and verifier (Spec
  0009 already points here).

## Purpose

The Spec 0009 runner writes only a trivial single-patch deterministic dump
(`_write_reconstruction_sample` -> `artifacts/reconstruction_samples.pt`, one patch,
zero epsilon; `src/eqvae/training/selected_runtime_runner.py:4023-4056`) and
clean/denoising validation rows (`metrics/validation_metrics.csv`). It does not
archive the canonical fixed 25 validation patches, save per-boundary reconstruction
progress, compare rotated-input against rotated-embedding reconstructions, persist
latent/embedding arrays or PCA maps, or emit `equivariance_error_25_patches` rows.

This spec specifies the replacement so that a run produces the issue-required
embedding-inspection artifacts, mirroring the FSQ reference evaluation loop and the
EQ-VAE paper's equivariance protocol, for the **non-equivariant baseline**, in a form
the future `SO(2)` model reuses unchanged:

- fixed 25 original patches plus per-half-epoch reconstruction progress;
- rotated-input reconstructions at `{0, 90, 180, 270}` degrees via exact
  `torch.rot90` (`D(E(rot90 x))`);
- rotated-embedding reconstructions from the deterministic posterior mean `mu`,
  rotated with the same exact `torch.rot90` (`D(rot90 E(x))`);
- full-frame error/difference maps;
- latent/embedding arrays plus EQ-VAE-style top-principal-component latent
  visualizations (plus a first-3-channels fallback);
- an `n = 25` equivariance metrics CSV whose headline
  `equivariance_error_25_patches` is the EQ-VAE / FSQ normalized latent L2-squared
  error ratio, feeding the training dashboard equivariance panel;
- manifest metadata (rotation method, angles) shared by both the image-rotation and
  embedding-rotation paths.

It is grounded in `GOAL.md:70-79`, `docs/repo_goal_and_requirements.md:49-56`,
`docs/repo_goal_and_requirements.md:75-91,109-149`,
`docs/issue_image_inventory.md:13-16,23-46`, Spec 0001's fixed-25 selector and
"Rotated And Latent Artifact Protocol"
(`docs/specs/0001-translatable-normal-vae-baseline.md:2937-2960,3027-3048`), Spec
0009's remaining fixed-25 blocker
(`docs/specs/0009-first-full-selected-runtime-training-run.md:389-401`), the FSQ
reference evaluation loop (`kaggle/train_runs:825-835,1010-1122`), EQ-VAE
(arXiv:2502.09509 and its code `github.com/zelaki/eqvae`), and decision
`docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`.

### Issue images inspected (2026-07-01)

Fetched and viewed the four inventory URLs
(`docs/issue_image_inventory.md:13-16`); all were fetchable.

- #3 and #4 dashboards are byte-identical: a six-panel grid (Composite Objective,
  Charbonnier L1, Structural Similarity Offset `1-SSIM`, PSNR dB, **Bottleneck
  Spatial Equivariance**, Learning Rate). The equivariance panel legend is
  **"Equivariance Error (25 Samples)"** plotted as an **"L2 Error Ratio"** on a **log
  scale** against a half-epoch x-axis. This fixes the headline metric: a normalized
  latent/bottleneck L2 error ratio over the 25 fixed patches, logged per half-epoch
  boundary.
- #4 grid: rows `0/90/180/270` degrees, columns **Ground Truth**, **Rotated Input
  Reconstruction**, **Rotated Latent Reconstruction**. The `0` row leaves the latent
  column blank (identity). For non-zero angles the rotated-latent column visibly
  degrades for the non-equivariant model; the same grid on the `SO(2)` model is
  expected to degrade less. The angle rows are exactly the `90`-degree multiples this
  spec uses.
- #4 PCA: four image columns (`SD-VAE`, `+Ours`, `SDXL-VAE`, `+Ours`) over four rows;
  each cell is a top-principal-component RGB projection of the latent map. The
  `+Ours` (more equivariant) latents look smoother. Our adaptation compares the
  baseline vs the `SO(2)` model with one shared PCA pipeline.

### EQ-VAE paper alignment (arXiv:2502.09509, code github.com/zelaki/eqvae)

The advisor asked (GitHub issues) to reuse the paper's inspection idea. The PDF/code
are external sources (not committed to the repo); the facts below were read this
window and should be re-verified before implementation. The `rot90` choice's primary
in-repo grounding is the FSQ reference (`kaggle/train_runs:1056-1068`) and the
issue-#4 grid image (rows `{0, 90, 180, 270}`). The paper corroborates:

- §3.3 Transformation Design: rotations are sampled from `{pi/2, pi, 3pi/2}`, i.e.
  `{90, 180, 270}` degrees; the paper's stated reason (§3.3, p.5) is that these are
  "multiples of 90" chosen "to avoid corner artifacts". (The paper also uses scale
  transforms `{0.25, 0.5, 0.75}`; scale is out of scope — see Non-Goals.)
- Appendix C.3 Equivariance Error:
  `Delta_eq = (1/(|T|*N)) * sum_T sum_N || tau . E(x) - E(tau . x) ||_2^2 /
  || E(tau . x) ||_2^2`, a normalized latent-space L2-squared error ratio over
  transforms `T` and `N = 50000` samples. This spec uses the same formula with
  `N = 25` and the `{90, 180, 270}`-degree rotation set — identical to the FSQ
  notebook metric (`kaggle/train_runs:1072-1075,1097`).
- Fig 2 / Fig 3: reconstruction under latent transformation compares `D(tau . E(x))`
  (rotated-embedding reconstruction) against `D(E(tau . x))` (rotated-input
  reconstruction) — the issue-#4 grid columns.
- Fig 1 / Table 5 + code `evaluation/vis_latent.py` (`pca_to_rgb`): top-3 principal
  components of the latent as the EQ-VAE latent visualization (see PCA definition).

## Non-Goals

- No implementation, runner, config, test, or verifier code in this window; this spec
  must be approved first (window instruction; spec-first rule `AGENTS.md:87-89`).
- No training-time change of any kind. This protocol adds no loss term, no
  augmentation, and does not alter the Spec 0009 training objective, runtime
  selection, DDP topology, or resume semantics. It is additive evaluation and reuses
  the committed durability primitives (`8d86f6a`).
- No FSQ quantization, codebook, discrete-index, PixelShuffle, or final-`tanh`
  artifacts. Use continuous-latent statistics only. Anchors:
  FSQ/codebook/discrete-latent `AGENTS.md:14-22`; FSQ + sub-pixel/PixelShuffle
  `GOAL.md:90-96`; no final `tanh` `GOAL.md:85` and
  `docs/decisions/0006-no-final-tanh-output.md`. The FSQ reference is reused only for
  its evaluation-loop structure and the equivariance metric, not its discrete latent
  artifacts.
- No continuous-angle rotation in this shared, comparable proxy. The evaluation uses
  exact `torch.rot90` at `{0, 90, 180, 270}` degrees (decision 0009). This is the
  evaluation methodology, not a change to the model's symmetry: continuous `SO(2)` is
  the equivariant model's conv architecture
  (`docs/decisions/0001-continuous-so2-scope.md`). Continuous-angle inspection for the
  `SO(2)` model (whose steerable convs make arbitrary-angle rotation meaningful) is a
  possible future extension, not part of this spec (`docs/issue_image_inventory.md:38`,
  "continuous angles for the `SO(2)` model").
- No nontrivial `SO(2)` irrep-aware latent-field transform. The baseline latent is a
  scalar spatial field (`LATENT_CHANNELS = 16`, `image_size // 8 = 32`, so
  `16 x 32 x 32`; `src/eqvae/models/non_equivariant_vae.py:19`). Only exact spatial
  `rot90` of the latent map is used; irrep-aware `mu`/`logvar`/sampling/KL actions
  belong to the `SO(2)` model
  (`docs/specs/0001-translatable-normal-vae-baseline.md:3046-3048`).
- No scale-equivariance (EQ-VAE `{0.25, 0.5, 0.75}`) and no LPIPS. This repo's target
  symmetry is continuous `SO(2)` rotation
  (`docs/decisions/0001-continuous-so2-scope.md`); scale is a possible future ablation
  and LPIPS is deferred to the later full-evaluator spec (user decisions, 2026-07-01).
- No new full Kaggle run, push, status read, output download, or resume. This window
  takes no remote action at all (window instruction). AGENTS.md separately permits
  Kaggle remote writes only with explicit permission and `KAGGLE_PUSH_CONFIRMED=1`
  (`AGENTS.md:103-106`); this is stricter than that.
- No full-dataset boxplots or metric summary tables over the whole validation/test
  set, and no sealed masked-WSI test evaluation. Those are later evaluator work
  (`docs/repo_goal_and_requirements.md:191-207`, FU-024/025).
- No final rendered six-panel dashboard figure or paper figure compositing. This spec
  produces the equivariance-curve **data** and the grid/PCA **image artifacts**;
  assembling `paper/sipaim2026/figures/*` is downstream paper work (FU-024).
- No paper claims, metric tables, or issue closure from these artifacts alone.

## Inputs And Data Contract

### Canonical fixed-25 selector (fail-closed, shared by both models)

The 25 images are a **single frozen canonical selector from the validation set**,
used for **every evaluation of BOTH the baseline and the future `SO(2)` model**, so
all visual and numerical comparisons are on identical inputs (decision 0009 point 4).
The FSQ notebook likewise drew its 25 from the validation set
(`kaggle/train_runs:827-830`); our canonical per-label selector (5 per label,
digest-sorted) is stricter and more reproducible than "first 25 of the val loader".

- Path: `configs/spec0001/fixed_25_validation_patches.json`, schema
  `spec0001.fixed_selector.v1`
  (`docs/specs/0001-translatable-normal-vae-baseline.md:2942-2956,2979-2995`).
- Reuse the existing loader/validator
  (`src/eqvae/data/fixed_selectors.py:325` `load_fixed_selector_document`,
  `:368` `validate_fixed_selector_document`,
  `:34` `FIXED_SELECTOR_PLACEHOLDER_STATUS`).
- The protocol MUST fail closed (raise, do not resample) if any of:
  - the file is missing or the schema version is not `spec0001.fixed_selector.v1`;
  - `status == "requires_real_data_generation"` or `selectors` is empty (the tracked
    config now holds the REAL selector as of FU-041, 2026-07-02; this clause still
    fails closed on any placeholder, e.g. a synthetic one in a CPU test);
  - `expected_count != 25` or the realized selector count `!= 25`;
  - `expected_per_label != 5` or any label `0..4` does not have exactly 5 rows;
  - the document fails `validate_fixed_selector_document` (noncanonical rows).
- Never silently resample a different set
  (`docs/specs/0001-translatable-normal-vae-baseline.md:2958-2960,2992-2995`).
- Local synthetic selectors for CPU tests must live under ignored `runs/` and MUST
  NOT overwrite the canonical tracked config
  (`docs/specs/0001-translatable-normal-vae-baseline.md:2974-2977,2990-2991`).

### Data and model

- Real data: `maximusshtefan/patches-pre-shuffled-ubc-ocean` validation shard
  `dataset/ubc_ocean_valid.{bin,csv}` (30000 patches), patch `3 x 256 x 256` CHW
  `uint8` normalized through the existing repo path (`normalize_uint8_batch`,
  `clean_validation_passthrough`;
  `docs/specs/0009-first-full-selected-runtime-training-run.md:80-86`).
- The 25 fixed patches are loaded directly and deterministically from the selector's
  per-row identity (`sample_id`/`wsi_id`/`x`/`y`), NOT drawn from the DDP sampler
  stream. All 25 are evaluated as one canonical global set, independent of
  `world_size`/rank (contrast the rank-0-local-shard hazard in FU-008).
- Model: `NonEquivariantVAE` (`src/eqvae/models/non_equivariant_vae.py:164`), with
  `encode(x) -> (mu, logvar)` (`:299`), `decode(latent) -> raw normalized RGB`
  (`:312`), and `forward(x, *, eps) -> VaeForwardOutput(reconstruction, mu, logvar, z,
  eps, ...)` (`:326-353`).
- Embedding-transform artifacts use the deterministic posterior mean `mu` from
  `encode`, never sampled `z`
  (`docs/specs/0001-translatable-normal-vae-baseline.md:3035-3036`).

### Rotation convention (single source, shared by both paths)

One rotation operator convention is defined once and applied to BOTH the image
(`256 x 256`) and the latent map (`32 x 32`):

- rotations are exact `torch.rot90(t, k, dims=[2, 3])` for `k in {0, 1, 2, 3}`, giving
  `{0, 90, 180, 270}` degrees, matching the FSQ reference
  (`kaggle/train_runs:1056-1068`) and EQ-VAE §3.3;
- `rot90` is a spatial permutation: exact, no interpolation, mapping the square frame
  onto itself, so there is no rotation-induced invalid region for either the image or
  the latent;
- the SAME `k` is applied to the image and to the latent map, so the two paths are
  directly comparable;
- because `rot90` is exact, there are no interpolation/padding/`align_corners`
  parameters; the manifest records `rotation.method = "rot90"` and the `k` set.

### Error/difference maps: full-frame, no mask

Error/difference maps are **full-frame (unmasked)**. `rot90` introduces no
rotation-invalid region, and the UBC patches carry no per-patch tissue mask, so there
is no masked variant (decision recorded 2026-07-01; matches FSQ, which used no mask).

### Angles and seeds

- The angle set is fixed: `{0, 90, 180, 270}` degrees (`k in {0, 1, 2, 3}`). `0`
  degrees is the identity reference used for originals, clean reconstruction progress,
  and the grid's first row; measured equivariance rows are for `k in {1, 2, 3}`
  (`{90, 180, 270}` degrees), matching FSQ and EQ-VAE.
- The `mu` path is deterministic. Sampled-latent equivariance uses a dedicated seeded
  generator and a paired-epsilon rule (below); the seed is recorded.

## Outputs And Acceptance Artifacts

Run-local outputs under the existing run `output_dir`
(`src/eqvae/training/selected_runtime_runner.py:1615-1633`). All directory writes are
atomic (write to a temp sibling, then rename — the FSQ `_tmp` -> final pattern,
`kaggle/train_runs:1017-1090`), and rank-0-only, reusing
`_write_json_atomic`/`_write_csv_atomic` (`selected_runtime_runner.py:3311,3323`). All
per-boundary artifacts are written every half-epoch boundary (FSQ-style full save).
Sizing (image-domain reconstruction/error-map tensors are stored `float16`, latents
`float32`, to bound growth): ~170 MB per boundary, so a 10-epoch / 20-boundary run
accumulates **~3.4 GB** of `.pt` tensors (plus small PNGs and the one-time
`originals.pt`). This fits Kaggle's ~19.5 GB working-output limit and is retained for
all boundaries by default; pruning (below) is the fallback only if a denser/longer run
binds the cap.

### Artifact tree (new `artifacts/fixed25/`)

- `artifacts/fixed25/manifest.json` — all metadata (see Config/Manifest below).
- `artifacts/fixed25/originals.pt` and a PNG montage — the 25 clean originals in image
  domain, each tagged with selector identity (`sample_id`, `wsi_id`, `label`, `x`,
  `y`); written once (the FSQ "immutable structural baseline",
  `kaggle/train_runs:825-835`).
- `artifacts/fixed25/boundary_{N:06d}/` — per half-epoch boundary `N`:
  - `reconstruction_progress.pt` (+ montage) — clean reconstruction of the 25
    (deterministic `mu` decode); satisfies per-boundary progress (`GOAL.md:73`,
    `docs/repo_goal_and_requirements.md:52`);
  - `rotated_angle_{deg}.pt` — per angle `deg in {90, 180, 270}`: rotated ground truth
    `rot90_k(x)`, rotated-input reconstruction `D(mu(rot90_k x))`, and
    rotated-embedding reconstruction `D(rot90_k mu(x))`;
  - `error_maps_angle_{deg}.pt` — full-frame difference maps for rotated-input-recon
    vs rotated ground truth and for rotated-embedding-recon vs rotated-input-recon;
  - `latent_mu.pt` — `mu` latent maps (`25 x 16 x 32 x 32`) for the clean originals,
    plus `rot90_k mu(x)` and `mu(rot90_k x)` per angle;
  - `grids/rotated_input_vs_latent_grid.png` — the issue-#4 grid (rows
    `{0, 90, 180, 270}`; columns GT, rotated-input recon, rotated-embedding recon);
    path-compatible with `paper/sipaim2026/figures/rotated_input_vs_latent_grid.*`
    (`docs/repo_goal_and_requirements.md:80-83`);
  - `latent_pca_eqvae_style.png` and `latent_first3.png` — the EQ-VAE-style top-3-PCA
    RGB latent visualization and the first-3-channels fallback
    (`docs/repo_goal_and_requirements.md:84-87,134-149`; EQ-VAE Fig 1 / `pca_to_rgb`).

This baseline run produces the baseline panels with the shared pipeline; the required
baseline-vs-`SO(2)` side-by-side composition (`docs/issue_image_inventory.md:40-46`)
is assembled downstream once the `SO(2)` model exists. The trivial
`_write_reconstruction_sample` dump is retired from the full run (superseded by the
above); the full-run verifier is updated accordingly, and the debug/tiny gate keeps
its trivial dump unchanged.

### Metrics

- `metrics/equivariance_25.csv` — one row per `(optimizer_step, angle_degrees,
  metric_name)`, with a `value` column plus `n` (= 25) and `mean`/`std` describing the
  25-patch distribution. Angles are `{90, 180, 270}` (`0` is the identity reference).
  All metrics are written at every half-epoch boundary. Metrics (all over the 25 fixed
  patches; see Architecture for exact formulas):
  - `equivariance_error_25_patches` — headline, the EQ-VAE / FSQ normalized latent
    L2-squared error ratio (matches the dashboard "Bottleneck Spatial Equivariance /
    L2 Error Ratio / 25 Samples" and EQ-VAE Appendix C.3);
  - `latent_mu_equivariance_error` — absolute latent `mu` equivariance error;
  - `latent_logvar_equivariance_error` — absolute clamped-`logvar` equivariance error,
    covering `docs/repo_goal_and_requirements.md:113`;
  - `reconstruction_equivariance_error` — image-domain rotated-embedding-recon vs
    rotated-input-recon error (full-frame);
  - `sampled_latent_equivariance_error` — paired/controlled-epsilon sampled
    equivariance (well-defined at all three measured angles because `rot90` is a
    permutation; see Metric definitions);
  - `rot90_exactness_error` — a sanity check that `rot90` is exact (must be `0`; see
    Metric definitions).
- The headline `equivariance_error_25_patches` is written every boundary as three
  per-angle rows (`angle_degrees in {90, 180, 270}`, each a genuine per-25-patch
  reduction with `value = mean_x r_k`), so every persisted row keeps the uniform
  `value`/`mean`/`std`/`n = 25` schema; the dashboard equivariance curve is their
  per-step mean (`docs/issue_image_inventory.md:23-29`).

### Verifier and manifest coverage

- Extend `verify_selected_runtime_full_output`
  (`src/eqvae/benchmarking/selected_runtime_gate.py`, CLI
  `eqvae.cli.selected_runtime_gate --verify-full-output`) and the `artifact_manifest`
  (`selected_runtime_runner.py:4783-4842`) to require the fixed-25 artifacts and drop
  the retired `reconstruction_samples.pt` check for the full run, so a run that omits
  the fixed-25 artifacts fails strict verification with explicit blockers (e.g.
  `selected_runtime_full_output_fixed25_originals_missing`,
  `..._equivariance_csv_missing`, `..._manifest_rotation_mismatch`). The debug/tiny
  verifier path is unchanged.

### Promotability label

Every fixed-25 artifact and the manifest carry a `data_source` field
(`real` | `synthetic`) and a `promotable` boolean. Artifacts built from a synthetic
selector or non-real data are `promotable = false` and must never be presented as
issue #4/#6 evidence
(`docs/specs/0009-first-full-selected-runtime-training-run.md:399-401`).

## Related Requirements And Evidence

- GitHub issues: #3 (metrics/dashboard), #4 (fixed-25 reconstructions, rotated grid,
  EQ-VAE latent viz), #6 (equivariant validation)
  (`docs/repo_goal_and_requirements.md:44-60`).
- Issue images: dashboard, rotated grid, EQ-VAE PCA (inspected above).
- Paper + code: EQ-VAE, arXiv:2502.09509 (§3.3, Appendix C.3, Fig 1/2/3),
  `github.com/zelaki/eqvae` (`evaluation/vis_latent.py` PCA) — the advisor asked to
  reuse the inspection idea.
- Reference: FSQ evaluation loop and `equivariance_error_25_patches`
  (`kaggle/train_runs:825-835,1010-1122`).
- Decisions: `0001-continuous-so2-scope`, `0002-normal-vae-baseline`,
  `0006-no-final-tanh-output`, `0009-fixed25-embedding-equivariance-eval-proxy`
  (`docs/decisions/README.md`).
- Paper artifacts: `reconstructions_25.*`, `rotated_reconstructions_25.*`,
  `rotated_input_vs_latent_grid.*`, `latent_pca_eqvae_style.*`,
  `equivariance_summary.tex` (`docs/repo_goal_and_requirements.md:75-91`).
- Related: Spec 0009 fixed-25 requirement
  (`docs/specs/0009-first-full-selected-runtime-training-run.md:389-401`).

## Architecture Or Workflow Contract

### Rotation operator

For `k in {0, 1, 2, 3}` (angle `90*k` degrees), define
`R_k(t) = torch.rot90(t, k, dims=[2, 3])`, applied identically to the image `x`
(`3 x 256 x 256`) and to the latent map (`16 x 32 x 32`). `R_k` is an exact spatial
permutation: invertible by `R_{(4-k) % 4}`, commutes with any elementwise operation,
and introduces no interpolation error.

### Metric definitions

For a clean patch `x`, let `mu(x) = encode_mu(x)` (the `16 x 32 x 32` posterior mean),
`logvar_c(x) = clamp_logvar(logvar(x))`
(`src/eqvae/models/non_equivariant_vae.py:404`), and `D` the decoder. Measured metrics
use `k in {1, 2, 3}` and are full-frame (no mask). Reductions are over the 25 patches;
report `value` = the 25-patch mean, plus `mean`/`std` and `n = 25`.

- `equivariance_error_25_patches` (headline, latent, normalized — EQ-VAE Appendix C.3
  / FSQ `kaggle/train_runs:1072-1075`): per patch and angle,
  `r_k(x) = || R_k(mu(x)) - mu(R_k x) ||_2^2 / (|| mu(R_k x) ||_2^2 + eps)` with
  `eps = 1e-8` (whole-`16x32x32`-tensor L2-squared). Each per-angle row `value` is
  `mean_x r_k`; no mean-over-angles row is persisted — the dashboard curve consumes
  the per-step mean of the three per-angle rows, matching FSQ
  `sum(eq_errors)/len(eq_errors)` (`kaggle/train_runs:1097`), so every persisted row
  stays a genuine per-25 reduction.
- `latent_mu_equivariance_error`: `mean_x || R_k mu(x) - mu(R_k x) ||` (absolute L2
  norm; one column).
- `latent_logvar_equivariance_error`:
  `mean_x || R_k logvar_c(x) - logvar_c(R_k x) ||`. The baseline `logvar` is a scalar
  `16 x 32 x 32` field, so its equivariance error is computable exactly like
  `latent_mu_equivariance_error`; covers `docs/repo_goal_and_requirements.md:113`.
  Posterior `mu`/`logvar` summary statistics
  (`docs/repo_goal_and_requirements.md:107`) are already emitted by the Spec 0001/0009
  `metrics/validation_metrics.csv` and are not duplicated here.
- `reconstruction_equivariance_error` (image domain, the grid comparison):
  `mean_x || D(R_k mu(x)) - D(mu(R_k x)) ||` (full-frame).
- `sampled_latent_equivariance_error` (paired epsilon): draw one `eps`
  (`16 x 32 x 32`) from the seeded generator; with
  `z(x, eps) = mu(x) + exp(0.5 * logvar_c(x)) . eps`, compare `D(R_k z(x, eps))`
  against `D(z(R_k x, R_k eps))` using the same underlying `eps`. Because `R_k` is a
  spatial permutation it commutes exactly with the elementwise product
  `exp(0.5 logvar) . eps`, so this metric is clean at all of `{90, 180, 270}` degrees
  with no interpolation/product-mixing floor. The paired-epsilon rule and seed are
  recorded (`docs/repo_goal_and_requirements.md:114`).
- `rot90_exactness_error` (sanity check): `mean_x || R_{(4-k) % 4}(R_k x) - x ||` in
  both image and latent domains. It must be `0` up to floating point; a nonzero value
  means the rotation was not implemented as exact `rot90` and must fail
  tests/verification.

At `k = 0` every equivariance error and the exactness error are `0`; a nonzero value
at `k = 0` indicates a convention bug and must fail tests.

### CSV schema note

All metrics are genuine per-25-patch reductions (including the headline, whose
per-patch quantity is `r_k(x)` at a fixed angle), so `value`/`mean`/`std`/`n = 25` are
well defined for every row. `value` carries the 25-patch mean; `mean`/`std` describe
the 25 per-patch values.

### PCA definition (from EQ-VAE `evaluation/vis_latent.py`)

For a latent map `L` of shape `[B, C, H, W]` (`C = 16`, `H = W = 32`), produce two RGB
images, matching the EQ-VAE reference `pca_to_rgb` and `first3_to_rgb`:

- `pca_to_rgb`: flatten to `X = L.permute(0,2,3,1).reshape(-1, C)` (`float32`); center
  `Xc = X - X.mean(0)`; covariance `cov = Xc.T @ Xc / (N - 1)`;
  `eigvals, eigvecs = torch.linalg.eigh(cov)`; take the top-3 eigenvectors
  `eigvecs[:, -3:]`; project `Xc @ top3` -> `[N, 3]`; then reshape channel-last as
  `reshape(B, H, W, 3).permute(0, 3, 1, 2)` -> `[B, 3, H, W]` (NOT a bare
  `reshape(B, 3, H, W)`, which would scramble the spatial layout since `X` was
  flattened channel-last); then **per-image min-max normalize** to `[0, 1]`. PCA is
  fit **per image** (`B = 1` per
  fixed patch), over that image's `H*W` spatial positions, over the channel dimension.
  Eigenvector sign is unpinned (qualitative), matching the reference.
- `first3_to_rgb`: take channels `[:, :3]` and per-image min-max normalize (fallback).

Render for the 25 clean latents and the rotated-embedding latents. The upscaled RGB
images form the EQ-VAE-style figure.

### Placement in the run loop

This is evaluation, not training (decision 0009): it never touches the loss or the
optimizer. It runs at every half-epoch boundary within the existing boundary block
(`selected_runtime_runner.py:2615-2687`) so its artifacts flush with the same
two-phase durability as metrics/checkpoints, mirroring the FSQ evaluation loop
(`kaggle/train_runs:1010-1122`):

1. `_log_full_boundary_start` breadcrumb (existing, `:2616`).
2. `_run_scheduled_validation` (existing, `:2621`).
3. Under `torch.no_grad`, run the full fixed-25 protocol for this boundary:
   reconstruction progress, rotated-input/rotated-embedding reconstructions, error
   maps, latent arrays, PCA/first3 images, the composite grid, and the equivariance
   CSV rows (three per-angle headline rows plus the other metrics).
4. First interval flush BEFORE checkpoint exposure (existing, `:2634`), now also
   writing the fixed-25 progress + equivariance CSV atomically.
5. Boundary checkpoints (existing, `:2649`).
6. Second interval flush refreshing artifacts with checkpoint hashes (existing,
   `:2668`).
7. `_synchronize_full_boundary_completion` DDP barrier (existing, `:2683`; the FSQ
   "Synchronization Point 2", `kaggle/train_runs:1119-1122`).

Because the protocol is model-agnostic and eval-only, the same evaluator also runs as
a standalone entry point over any `best_model.pt`/`final.pt` and is reused unchanged
for the `SO(2)` model.

### DDP and resume

- All fixed-25 computation and writing happen on the primary rank (`_is_primary_rank`,
  `:1782`); other ranks wait at the existing barrier (`_barrier`, `:1786`). The 25
  patches are loaded as one canonical global set, not a per-rank shard, so the result
  is `world_size`-independent (avoids the FU-008 rank-0-shard bug). This matches the
  FSQ eval, which runs the whole protocol under `if local_rank == 0`
  (`kaggle/train_runs:1027`).
- Because the fixed-25 rows are already global and rank-0-produced, they MUST NOT be
  routed through `_gather_csv_rows` (`:1792`), which concatenates every rank's rows
  and is correct only for the per-rank-tagged train/validation CSVs. Gathering a
  global row would either duplicate it `world_size` times (violating the
  `world_size`-independence criterion) or force a symmetric collective around
  rank-0-only compute and risk a hang. These rows are merged, not gathered.
- Durability is NOT automatic for a new CSV. Resume-durability is only needed for the
  per-boundary equivariance CSV time-series (so the dashboard curve survives a
  cancel+resume, the FU-039 lesson); the per-boundary image/latent snapshots are
  regenerated each boundary and need no history. The runner has no generic durable-CSV
  mechanism; gather-then-merge and resume-prefix durability are wired per-CSV, once
  each for `train_steps` and `validation_metrics`. For `metrics/equivariance_25.csv`
  the implementation MUST add, mirroring those two: (a) a `_RunArtifacts` path field
  (`:338-351`); (b) a `_read_resume_csv_prefix` call plus a `resume_equivariance_rows`
  field in the resume-history load (`_load_resume_artifact_history`, ~`:1827-1840`),
  keyed on `optimizer_step <= start_step` as validation does; (c) a full-run
  resume-prefix validation for the CSV; (d) an accumulator threaded through
  `_interval_flush_state`/`_IntervalFlushState`; and (e) a `_write_csv_atomic` call in
  `_write_partial_interval_artifacts` (`:3025`). Each boundary flush rewrites the whole
  file, so the resume prefix must be re-prepended once via `_merge_resume_csv_rows`
  (`:1910`, prefix applied once after any gather, per the `8d86f6a` fix). Without
  steps (b)-(e) the atomic temp-then-rename overwrite (`_write_csv_atomic`, `:3323`)
  silently truncates all pre-resume equivariance rows — the exact FU-039 missing-prefix
  bug class. (If the project chooses restart-from-scratch over resume for the full run,
  the CSV needs no resume-merge; but the durable wiring is the safe default.)
- Rank-0 write failure is broadcast so all ranks raise together
  (`_broadcast_rank0_error`, `:2971`).

## Config Contract

Add a shared `fixed25_equivariance` object in
`configs/spec0001/non_eq_vae_model_base.json` (enabled by
`configs/spec0001/non_eq_vae_selected_runtime_full.json`), so the baseline and the
future `SO(2)` model use byte-identical evaluation config:

- `enabled` (bool);
- `selector_config = "configs/spec0001/fixed_25_validation_patches.json"`;
- `expected_count = 25`, `expected_per_label = 5`;
- `rotation.method = "rot90"`, `rotation.dims = [2, 3]`,
  `rotation.k_values = [0, 1, 2, 3]` (angles `[0, 90, 180, 270]`; measured metrics use
  `k in {1, 2, 3}`);
- `latent.transform = "rot90_scalar_field"`,
  `latent.source = "posterior_mu_deterministic"`, `latent.channels = 16`,
  `latent.spatial = 32`;
- `sampled_latent.paired_epsilon = true`, `sampled_latent.epsilon_seed` (int);
- `equivariance.error_eps = 1e-8` (matches FSQ `kaggle/train_runs:1074`);
- `save_every_boundary = true` (full FSQ-style save each boundary);
- `pca.methods = ["pca_top3", "first3"]`, `pca.components = 3`,
  `pca.fit_scope = "per_image"`, `pca.sign_convention = "unpinned"`;
- `error_maps.masked = false` (full-frame only);
- `promotable_requires_real_data = true`.

The manifest (`artifacts/fixed25/manifest.json`) records the resolved values of all of
the above, plus: selector path/sha256/schema/status/per-patch identities;
`rot90_exactness_error` (must be `0`); `data_source` (`real`/`synthetic`) and
`promotable`; and the boundary `optimizer_step` list covered. `rotation.*` is stored
once and referenced by both the image and latent paths so the verifier can assert a
single shared convention.

## Acceptance Criteria

1. The protocol fails closed (raises, no resample) when the selector is missing, is
   the `requires_real_data_generation` placeholder, has empty `selectors`, has count
   `!= 25`, has any label without exactly 5 rows, or fails
   `validate_fixed_selector_document`.
2. A synthetic/local selector is never written to the canonical tracked config;
   synthetic artifacts are `promotable = false`.
3. The identical frozen canonical fixed-25 validation selector is the only source of
   the 25 images, so a later `SO(2)`-model evaluation uses the same 25 inputs.
4. The 25 clean originals are archived once with selector identity metadata; clean
   reconstruction progress for the 25 is written at every half-epoch boundary.
5. Rotated-input and rotated-embedding reconstructions are produced at every boundary
   for every angle in `{90, 180, 270}` using exact `torch.rot90` (same `k` for image
   and latent); the embedding-rotation path uses deterministic `mu`, never sampled
   `z`.
6. Full-frame error/difference maps are produced (no mask).
7. Latent `mu` arrays, the EQ-VAE-style top-3-PCA visualization, and the
   first-3-channels fallback are produced.
8. `metrics/equivariance_25.csv` includes `equivariance_error_25_patches` plus the
   reconstruction, latent-`mu`, latent-`logvar`, sampled-latent, and `rot90_exactness`
   metrics, each a required column that strict verification and the CSV-column test
   check for. Each metric is emitted at every boundary with `n = 25`, per measured
   angle; the headline metric is three per-angle rows per boundary.
9. `rot90_exactness_error` is `0` up to floating point, and at `k = 0` all
   equivariance errors are `0`; a nonzero value fails tests/verification.
10. The manifest records the rotation method (`rot90`), `k` values, and angles once,
    shared by both paths, plus the promotability label.
11. Artifact writes are atomic and rank-0-only. The canonical fixed-25 rows are merged
    with the resume prefix (`_merge_resume_csv_rows`) but NOT gathered across ranks;
    the equivariance CSV has its own resume-prefix read plus accumulator so pre-resume
    rows survive resume; the rank-0 failure broadcast and the final barrier are
    respected. Results are `world_size`-independent for the canonical 25.
12. The protocol adds no loss term, no augmentation, and does not change the training
    objective (evaluation-only, decision 0009).
13. The full-run verifier requires the fixed-25 artifacts and no longer requires the
    retired `reconstruction_samples.pt`; the debug/tiny verifier path is unchanged.
14. `./scripts/python_quality.sh` passes (Ruff `ALL`, BasedPyright strict, no new
    ignores) and focused CPU-only tests pass.
15. Adversarial subagent review finds no high-severity blocker before implementation
    is called done.

## Tests And Verification Commands

Focused CPU-only tests (new `tests/test_fixed25_equivariance_artifacts.py` plus
extensions to `tests/test_selected_runtime_full_run.py`):

- fail-closed on the committed placeholder selector and on a count-`!= 25` synthetic
  selector;
- a synthetic 25-patch selector under `runs/` drives a tiny CPU model to produce, at
  each simulated boundary, originals, reconstruction progress, rotated grids (correct
  shapes at `{90, 180, 270}`), full-frame error maps, latent arrays, the PCA and
  first3 images, and the equivariance CSV;
- the embedding-rotation path reads `mu` (assert it is not the sampled `z` path);
- `rot90_exactness_error == 0` and `k = 0` gives zero equivariance error;
- image and latent paths share one rotation `k` read from the manifest;
- equivariance CSV has all required metric names (incl. latent-`logvar`) with
  `n = 25`;
- `pca_to_rgb` output shape `[1, 3, 32, 32]` in `[0, 1]` and reproducible for a fixed
  latent;
- a simulated `world_size = 2` boundary flush neither drops nor duplicates pre-resume
  equivariance rows;
- the extended verifier fails when a fixed-25 artifact/column/manifest field is removed
  and no longer requires `reconstruction_samples.pt` for the full run.

Commands (run only when implementation exists; raw pytest needs the repo-safe TMPDIR
per the window environment notes):

```bash
TMPDIR=/home/maximus/Documents/Tesis/.agent-tmp/equivariant-vae \
  PYTHONPATH=src .venv/bin/pytest tests/test_fixed25_equivariance_artifacts.py -q
TMPDIR=/home/maximus/Documents/Tesis/.agent-tmp/equivariant-vae \
  PYTHONPATH=src .venv/bin/pytest tests/test_selected_runtime_full_run.py -q
./scripts/python_quality.sh          # ~5 min; pass ~600000 ms timeout
git diff --check
./scripts/agent_preflight.sh
cd /home/maximus/Documents/Tesis && ./agent_preflight.sh
```

## Implementation Blockers

- Spec approval by the user (this window is spec-only).
- Paper-promotable artifacts require the REAL fixed-25 selector, which needs the real
  Kaggle validation CSV/bin not present locally
  (`docs/specs/0001-translatable-normal-vae-baseline.md:2962-2977`). So promotable
  fixed-25 artifacts are only produced during a real (Kaggle or real-data-mounted)
  run. Local implementation and tests use synthetic
  selectors and are non-promotable.

## Known Risks

- Over-reading the proxy. `rot90` probes the discrete `C4` subgroup; the equivariance
  error is a proxy for embedding smoothness/structure used to COMPARE the two models,
  not a full characterization of continuous `SO(2)` equivariance (which is the
  equivariant model's architectural property). Label artifacts as `C4`/`rot90` and
  present the metric comparatively, not as a pass/fail on the baseline.
- PCA basis differs per image and per model (each fit independently, sign unpinned),
  so colors are qualitative — the comparison is about smoothness/structure, not
  absolute color. This matches the EQ-VAE reference.
- Presenting synthetic/non-real artifacts as paper/issue evidence. Mitigation:
  `promotable`/`data_source` labels and the verifier.
- Comparing the two models on different images. Mitigation: the single frozen
  canonical validation selector shared by both models (Acceptance Criterion 3).
- Full-save-every-boundary disk growth. The artifacts total **~3.4 GB** for a 10-epoch
  / 20-boundary run (image tensors `float16`, latents `float32`, ~170 MB/boundary),
  which fits Kaggle's ~19.5 GB working-output limit but is materially larger than an
  earlier "< ~1 GB" estimate. Mitigation if a denser/longer run (or the reused `SO(2)`
  evaluation) binds the cap: prune old boundary directories keeping the latest few plus
  best/final (documented if applied; not yet implemented, since the current schedule
  fits).
- Coupling to the undecided v1-continuation and to a full run that has not been
  approved; this spec must not imply approval for any remote action.

## Adversarial Checks

A skeptical reviewer should try to:

- pass the committed `requires_real_data_generation` placeholder (or a 24/26-row or
  wrong-per-label selector) and confirm the protocol raises rather than resampling;
- overwrite `configs/spec0001/fixed_25_validation_patches.json` with a synthetic
  selector and confirm it is refused;
- confirm the baseline and a later `SO(2)` evaluation load byte-identical 25 images
  from the same canonical selector;
- swap the embedding-rotation path to sampled `z` and confirm a test fails;
- apply a different `k` to the image than to the latent and confirm the manifest check
  / test fails;
- feed `k = 0` and confirm zero equivariance error; confirm `rot90_exactness_error` is
  `0` and that an interpolated (non-`rot90`) rotation would make it nonzero and fail;
- run with `world_size = 2` and confirm the 25-patch result is identical to
  `world_size = 1` (no per-rank sharding) and that resumed boundaries do not duplicate
  or drop equivariance rows;
- remove any single fixed-25 artifact, CSV column, or manifest field and confirm strict
  full-output verification fails; confirm the full-run verifier no longer depends on
  `reconstruction_samples.pt`.

## Resolved Decisions And Remaining Notes

All prior open questions were resolved with the user on 2026-07-01 and are recorded in
"Settled implementation decisions" above (rotation = rot90; cadence = full save every
boundary; error maps = unmasked/full-frame; PCA = paper `pca_to_rgb` + `first3`;
retire the placeholder from the full run; scale/LPIPS out of scope; config home =
shared `model_base` block; placement = shared in-run + standalone evaluator; standalone
Spec 0010). Remaining, non-blocking implementation notes:

- Boundary-artifact retention if Kaggle output limits bind (default: keep all; prune
  oldest boundaries keeping best/final if needed, documented if applied).
- Whether the composite grid/PCA images are rendered in-run each boundary or only the
  arrays are saved in-run with images rendered by the standalone evaluator (either is
  acceptable; in-run rendering is cheap).

## Addendum: fixed-25 selector generation kernel (FU-041)

The canonical selector `configs/spec0001/fixed_25_validation_patches.json` must be
generated from the REAL UBC validation shard, which is not resolvable locally
(`resolve_patch_data_paths("auto")` only finds it under `/kaggle/input/...`). The
dedicated CPU kernel `kaggle/kernels/fixed25_selector` generates it where the dataset
is mounted, with byte-exact source provenance. This section authorizes that kernel
(guard token: `fixed25_selector_kernel_ready`).

**Status: DONE (2026-07-02).** The kernel ran on Kaggle, the selector was downloaded,
verified (`status: pass`, 25 selectors, 5 per label 0..4, validation split,
`crc_checked: true` from `ubc_ocean_valid.bin`), and committed to the tracked config;
the 25 images are frozen in `docs/data/fixed25/` (`originals.png` + lossless-uint8
`originals.pt`). The remainder of this section documents the (idempotent) generation
contract.

- **CPU only, no GPU.** The step reads the validation shard, digest-sorts 5-per-label
  (locked `FIXED_25_VALIDATION_SEED`), and writes JSON + 25 images; there is no model
  forward pass. `kernel-metadata.json` sets `enable_gpu: false` with no `machine_shape`,
  so it never consumes T4 quota. The push guard rejects a GPU shape.
- **What it runs** (both CLIs are already in the embedded payload; the config resolves
  `source_config` to `non_eq_vae_model_base.json`):
  1. `python -m eqvae.cli.select_fixed_patches --config
     configs/spec0001/non_eq_vae_selected_runtime_full.json --kind fixed_25_validation
     --data-root auto --output /kaggle/working/fixed_25_validation_patches.json
     --validate-crc` — generates the canonical selector to the working dir (NOT the
     tracked config path, so no `--allow-tracked-config-overwrite`).
  2. `python -m eqvae.cli.fixed25_originals --config <same> --data-root auto --selector
     /kaggle/working/fixed_25_validation_patches.json --output-dir /kaggle/working` —
     archives the 25 selected patches as `artifacts/fixed25/originals.pt` (images
     stored losslessly as uint8, reconstruct `x/255*2-1`) and a montage
     `originals.png` (no checkpoint needed), so the exact images are reviewable.
- **Validation-only, fail-closed.** `--kind fixed_25_validation` forces the validation
  split, and every fixed-25 load path additionally raises unless the selector's
  `source_split == "validation"`. The masked-holdout WSIs are already excluded from the
  validation shard, so no extra holdout filter applies (holdout filtering is
  train/`fixed_32`-only).
- **CRC contract (settled 2026-07-02).** The canonical selector is CRC-validated
  (`crc_checked=True`, honoring `canonical_overwrite_requires_crc`), and
  `validate_fixed_selector_document` compares `crc_checked` for equality, so the fixed-25
  load path validates the selector shard with CRC on BOTH sides — the standalone
  evaluator, the `fixed25_originals` CLI, and the full-run `_prepare_fixed25_runtime`
  all use `validate_crc=True` for the selector shard (independent of the general
  `data_surface.validate_crc`, which stays `False` for the training/validation loaders).
  This mirrors the fixed-32 readiness convention and prevents a real CRC-validated
  selector from failing closed in the run.
- **Guards.** The push is the generic guarded push
  (`KAGGLE_PUSH_CONFIRMED=1 KAGGLE_FULL_DATASET_CONFIRMED=1 ./scripts/kaggle_kernel.sh
  push kaggle/kernels/fixed25_selector`); `guard_fixed25_selector_push_ready` requires a
  clean worktree (`--verify-only` without `--allow-dirty`), the CPU/dataset metadata
  contract, this spec's `fixed25_selector_kernel_ready` token, and the required run.py
  literals. Remote status/output are separate `KAGGLE_REMOTE_CONFIRMED=1` reads
  (`status-fixed25-selector`, `output-fixed25-selector`). Local preflight:
  `./scripts/kaggle_kernel.sh preflight-fixed25-selector`.
- **Promotability.** The kernel output is generation evidence; the selector is
  paper-promotable once committed (done 2026-07-02, after user review of the JSON + the
  originals montage) and consumed by a real full run. The commit overwrote the former
  tracked placeholder (user-approved).

## Related Files

- `GOAL.md`
- `docs/repo_goal_and_requirements.md`
- `docs/issue_image_inventory.md`
- `docs/open_follow_ups.md`
- `docs/decisions/README.md`
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`
- `docs/specs/0001-translatable-normal-vae-baseline.md`
- `docs/specs/0009-first-full-selected-runtime-training-run.md`
- `kaggle/train_runs` (FSQ reference evaluation loop and equivariance metric)
- `src/eqvae/training/selected_runtime_runner.py`
- `src/eqvae/models/non_equivariant_vae.py`
- `src/eqvae/data/fixed_selectors.py`
- `src/eqvae/artifacts/` (to be populated; PCA per EQ-VAE `evaluation/vis_latent.py`)
- `src/eqvae/benchmarking/selected_runtime_gate.py`
- `src/eqvae/cli/selected_runtime_gate.py`
- `configs/spec0001/fixed_25_validation_patches.json`
- `configs/spec0001/non_eq_vae_model_base.json`
- `configs/spec0001/non_eq_vae_selected_runtime_full.json`
