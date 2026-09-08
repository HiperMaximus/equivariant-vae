# Spec 0040: Spatial Latent PCA and Local-Coherence Visualization

Status: implemented / locally verified
Owner/workstream: advisor-facing, local-only post-hoc visualization
Last updated: 2026-09-04

## Purpose

Add an EQ-VAE-style false-colour visualization of the frozen final posterior
mu[16,32,32], plus a quantitative local-coherence companion. The goal is to
show how latent descriptors vary across spatial positions, not to replace the
rotation-orbit or exact-quarter residual evidence from Spec 0038.

Add one exploratory pointwise RGB readout to answer a separate diagnostic
question raised by the PCA maps: can a single local 16-dimensional descriptor
linearly recover coarse source-patch colour without spatial mixing?

## Frozen Scope

- Read only the already SHA-pinned fixed-25 latent_mu.pt archives and canonical
  fixed-25 originals used by Spec 0038. Do not train, rerun a model, access
  Kaggle, or touch sealed test data.
- Use mu_clean[25,16,32,32] only. Preserve the existing comparable view: one
  PCA per model over all 25 times 616 central-disk descriptors. Add a second,
  explicitly visual-only paper-style view: one PCA per displayed patch and
  model over the full native 32x32 field.
- Predeclare display patch indices 0, 8, 12, and 24. All 25 fixed patches enter
  the PCA fit and local-coherence summary.
- The pointwise readout has one declared split only: fit on fixed patches 0--19
  and display held-out patches 20--24. It must never train on a displayed
  held-out patch, use the decoder, use neighbouring cells, access new data, or
  select a hyperparameter from the held-out results.

## Construction

For model m, image n, and disk position p, define the 16-dimensional local
latent descriptor:

    f_{m,n}(p) =
    (mu_{m,n,1}(p), ..., mu_{m,n,16}(p)) in R^{16}.

Stack every descriptor into
F_m in R^{(25 |D|) x 16}, centre it by its global mean, and retain the three
leading PCA directions W_m in R^{16 x 3}. For every displayed location:

    y_{m,n}(p) = (f_{m,n}(p) - bar f_m) W_m in R^3.

Map the three scores to RGB after symmetric clipping at one model-wide scale
s_m: the 99th percentile of abs(y_k) pooled over all three components and all
25 disk maps. Thus RGB_k is clip(1/2 + y_k/(2 s_m), 0, 1). This prevents a
low-variance PC3 from being given the same visual contrast as PC1. The PCA
basis and colour scaling are per model: colours describe each model's internal
spatial variation, not an alignment of separately trained channel semantics.
Record the sign convention (the largest-absolute component loading is positive)
and each of the first three explained-variance fractions.

### Paper-style per-patch view

The public EQ-VAE visualizer treats positions of one latent image as samples.
For a selected image n, form X_m,n in R^(1024 x 16) with the descriptor at
every native 32x32 location as its row. Centre X_m,n, take its three leading
channel-PCA directions V_m,n, and return the three scores to their original
locations:

    Z_m,n = (X_m,n - 1 bar_x_m,n^T) V_m,n in R^(1024 x 3).

Use one joint min/max over all three score planes of this one map:

    RGB_m,n = (Z_m,n - min(Z_m,n)) / (max(Z_m,n) - min(Z_m,n)).

Upsample that 32x32 RGB field bilinearly only for display. Do not blend it
with the source patch: show the original as a separate tile. This contrast
normalization is deliberately reproduced for visual intuition; hues, signs,
brightness, and magnitudes are not comparable between patches or models.
Do not apply the disk mask, sign stabilization, percentile clipping, or
relative-edge metric to this paper-style view.

### One-shot pointwise RGB probe

Downsample each canonical 256x256 RGB source patch to T_n in [0,1]^(3x32x32)
with antialiased bilinear interpolation. For each model separately, use all
locations of fixed patches 0--19 to fit one ridge-regularized affine map:

    yhat_m,n(p) = W_m f_m,n(p) + b_m,
    W_m in R^(3x16), b_m in R^3.

Features are standardized from the 20 training patches only. Use fixed ridge
lambda=1e-3 and solve the closed-form least-squares system once. Clip only the
rendered RGB prediction to [0,1]. The probe therefore has 51 learned numbers
per model, shared at every spatial location; it cannot reconstruct a colour
from adjacent latent cells or memorize a patch-specific spatial template.

For held-out patch n, score the un-clipped prediction against a constant RGB
baseline equal to the training-patch channel mean Tbar_train:

    R2_m,n = 1 - sum_(p,c) [yhat_m,n,c(p) - T_n,c(p)]^2
                   / sum_(p,c) [Tbar_train,c - T_n,c(p)]^2.

This is an out-of-sample generalization baseline: the readout and the baseline
are both fixed without seeing patches 20--24. The reported MSE is also
evaluated in RGB [0,1] before the display-only clipping step.

For every fixed patch, use the undirected horizontal/vertical disk-neighbour
set E to compute local relative edge RMS:

    r_m(n) =
       sqrt((1 / (16 |E|)) sum_c sum_(p,q in E)
            [mu_{m,n,c}(p) - mu_{m,n,c}(q)]^2)
       /
       sqrt((1 / (16 |D|)) sum_c sum_(p in D)
            [mu_{m,n,c}(p) - bar mu_{m,n,c}]^2).

The denominator is centred spatial RMS. If it is exactly zero, define r_m=0,
flag the map as degenerate, and retain unnormalised edge RMS, centred spatial
RMS, and posterior RMS. Lower r_m means less adjacent spatial variation
relative to the model's own spatial variation; it is not a general quality or
equivariance score.

## Outputs

- 04-spatial-latent-pca.png: four predeclared fixed-25 rows containing the
  input and normal/SO(2) latent PCA-RGB maps, plus a fixed-25 local-coherence
  summary on shared numeric axes.
- The existing offline rotation-orbits.html gains Section 5. Its SVG maps and
  all-25 paired normal/SO(2) summary are drawn from precomputed arrays rather
  than embedding the PNG.
- 05-paper-style-latent-pca.png shows the same four originals alongside the
  per-patch normal/SO(2) PCA-RGB maps. Section 5 adds the equivalent dynamic
  browser rendering and an implementation-level explanation of both PCA fits.
- 06-pointwise-rgb-probe.png shows the five held-out downsampled sources next
  to their normal/SO(2) pointwise predictions and reports held-out pixel R2
  against the training-colour-mean baseline.
- The offline rotation-orbits.html gains Section 6. It draws the same five
  held-out source/prediction rows from local numeric arrays, explains the
  training-only standardization and affine ridge objective, defines its R2
  baseline, and states why this is neither a decoder reconstruction nor an
  equivariance score.
- The Spec 0038 manifest gains fit scope, selected patch indices, PCA explained
  variance/score scales, and all 25 local-coherence values.

## Interpretation Contract

- The existing common-scale view is inspired by the EQ-VAE PCA-RGB convention,
  not a literal reproduction: it uses a shared all-25 disk fit per model,
  neutral excluded exterior, and nearest-neighbour display of native cells.
- The paper-style companion intentionally reproduces the released visual
  convention's per-image, whole-grid PCA, joint RGB min/max, and bilinear
  display. It is not a blend with the original image and is not numerical
  equivariance, smoothness, or cross-model evidence.
- A grainier PCA-RGB map means adjacent 16-dimensional descriptors vary more
  rapidly in the PCA directions. It is qualitative only.
- A signed arithmetic mean of differences is invalid as an error or roughness
  measure because positive and negative deviations cancel. Mean absolute
  difference is a valid alternative, but RMS measures Euclidean magnitude in
  the latent coordinate system, preserves latent units, and emphasizes isolated
  large changes. State this choice and its trade-off.
- Do not call PCA-map hue differences a cross-model comparison. Use the
  shared-formula local relative edge RMS only as its numerical companion.
- A positive held-out pointwise readout is evidence that local posterior
  descriptors retain coarse appearance information, not evidence of
  equivariance, reconstruction quality, or a better VAE. A negative result
  does not prove absence of information: it may be nonlinear or spatially
  distributed.
- Do not claim an SO(2) advantage unless the archived fixed-25 values support
  it. Report the actual normal/SO(2) distributions without directionally
  favourable language.

## Acceptance Criteria

- PCA accepts only finite [25,16,32,32] arrays and a matching disk mask; it
  returns three RGB channels, model-wide PCA explained variance, and
  reproducible scalar score scale. Rank-zero PCA has a finite zero fallback.
- The local metric is zero and flagged degenerate for a spatially constant
  field, is mathematically invariant to a common nonzero global scalar
  multiplier for nondegenerate fields (up to floating-point roundoff), and
  rejects an incompatible mask.
- All displayed RGB maps use the same per-model PCA fit and score scales;
  exterior disk cells are visibly marked as excluded.
- The paper-style function accepts a finite [N,16,32,32] array, produces one
  [32,32,3] RGB map per input field, uses a separate full-grid PCA and one
  joint three-channel min/max range per map, and has a finite constant-field
  fallback. Its renderer uses bilinear display and keeps the original patch in
  a separate column.
- The pointwise probe has exactly 20 training and 5 held-out fixed25 images,
  never overlaps them, fits only an affine 16-to-3 map plus bias per model, and
  recovers a known synthetic affine spatial target on held-out locations.
- The PNG, HTML section, manifest, and documentation include the exploratory
  label and fit/scale caveats. The HTML stays offline and has no raster
  dashboard.
- The all-25 comparison visibly preserves patch pairing and reports the actual
  paired result without claiming an SO(2) gain.
- Focused tests, touched-file Ruff/format, BasedPyright, JavaScript parse
  check, PNG inspection, archive-hash validation, git diff --check, and
  workspace preflight pass.

## Risks And Review Questions

- Separate encoder channel coordinates can be permuted or rotated, so a joint
  colour basis may falsely imply semantic colour alignment. Per-model PCA plus
  a channel-space metric is required.
- Low local edge RMS can mean a collapsed or blurred latent field. Raw spatial
  and posterior RMS must remain visible.
- PCA compression may hide variation outside its top three directions. Report
  the three-component explained-variance fraction.

## Implementation Record

- The common-scale diagnostic is implemented in
  `src/eqvae/artifacts/rotation_orbits.py` and its existing
  local renderer. `04-spatial-latent-pca.png`, Section 5 of
  `rotation-orbits.html`, and schema-v4 `manifest.json` are in ignored
  `runs/local/frozen_vae_rotation_orbits/`.
- The exact SHA-pinned archives gave median `r_edge` 1.456673 for normal and
  1.467611 for SO(2). The paired mean `SO(2)-normal` difference is +0.010646,
  positive on 23/25 patches. This metric therefore does not support a local
  SO(2) coherence advantage; the page says so explicitly.
- The paper-style companion is implemented in the same local renderer as
  `05-paper-style-latent-pca.png` and Section 5 of the HTML. It uses posterior
  means, not posterior samples, so it is deterministic; its four actual
  frozen-checkpoint maps remain visually grainy after the released method's
  per-tile fit, contrast, and bilinear display. That is a result about these
  posterior descriptors, not a rendering failure. Focused tests (14),
  touched-file Ruff/format, BasedPyright, JavaScript parse, manifest JSON,
  archive-hash validation, PNG inspection, and diff check passed. The broader
  repository quality gate remains blocked by unrelated pre-existing packaged
  Kaggle-kernel lint failures.
- The one-shot pointwise probe is implemented in the same renderer and emitted
  as `06-pointwise-rgb-probe.png`. It fit one 51-parameter affine 16-to-RGB
  ridge map separately for each frozen model on patches 0--19 and showed only
  patches 20--24. Mean held-out pixel R2 against the predeclared training-RGB
  mean baseline is `0.285982` for normal and `0.188932` for SO(2); mean RGB
  [0,1] MSE is `0.002657` and `0.003014`, respectively. The held-out panels
  preserve some coarse tissue layout but are visibly blurred; that is limited
  evidence of local linear appearance information, not an equivariance or VAE
  quality comparison. The synthetic affine recovery contract, source/type
  checks, and the 16 focused tests pass.
- The native Section 6 view keeps the source and the two predictions separate,
  shows the held-out R2/MSE next to every prediction, and makes the 0--19/20--24
  split visible. The paired r_edge HTML axis no longer overlays categorical
  model labels with numeric ticks; its reading guide now explicitly identifies
  a dot, a connector, the normalized vertical quantity, and the sign of the
  right-panel difference.

## Related Files

- docs/specs/0038-frozen-vae-rotation-orbit-visualization.md
- docs/specs/0010-fixed25-equivariance-artifact-protocol.md
- src/eqvae/artifacts/rotation_orbits.py
- src/eqvae/cli/render_frozen_vae_rotation_orbits.py
