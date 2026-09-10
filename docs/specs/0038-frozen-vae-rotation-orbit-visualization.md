# Spec 0038: Frozen VAE Rotation-Orbit Visualization

Status: implemented / dense trajectory superseded by Spec 0050
Owner/workstream: advisor-facing post-hoc visualization
Last updated: 2026-09-08

## Purpose

Create five compact PNG figures and one self-contained local HTML companion
that make the frozen normal-VAE and continuous-SO(2) VAE response to a smooth
input rotation visible on the established fixed validation 25. This is an
exploratory explanatory view, not a replacement for the Spec 0010 exact-`rot90`
protocol or a held-out evaluation.

## Frozen Scope

- Load only the accepted 60,000-update checkpoints through a hash-checked
  local checkpoint or exact state-dict bundle. Do not train, alter weights,
  access sealed test data, or change paper claims.
- Use canonical `docs/data/fixed25/originals.pt`, require it to be byte-identical
  to both stored run archives, and reconstruct model inputs as `x/255*2-1`.
- Sample angles `0,...,359` degrees. The displayed endpoint at 360 degrees is
  a copy of angle 0, never a separately interpolated inference.
- Inputs remain `3x256x256`: use the repository's existing inverse-sampling
  rotation matrix, bilinear `grid_sample`, zero padding, and
  `align_corners=False`; exact multiples of 90 use `torch.rot90`. All numerical
  spatial comparisons use a centered inscribed disk; exterior/boundary pixels
  are not evidence. Record the input round-trip interpolation floor.
- One separately authorized private Kaggle T4 run may evaluate all 25 fixed
  validation patches at every integer degree. It must attach only
  `maximshtefan/eqvae-vae-test-reconstruction-inputs-v1` for the exact frozen
  state dicts and `maximusshtefan/patches-pre-shuffled-ubc-ocean` for the
  validation binary, validate every weight and selected-patch hash, require
  CUDA, use FP32 inference, and keep internet disabled. Authorization marker:
  `spec0038_fixed25_dense360_population_authorized`.

## Outputs And Acceptance Artifacts

- A tracked renderer and focused CPU tests.
- Ignored local output directory containing:
  - `01-latent-orbits.png`: one predetermined fixed-25 PCA orbit example for
    each model and a documented all-25 masked exact-quarter residual summary.
  - `02-f1-phase.png`: trained and seeded-untrained SO(2) F1 mean-vector
    diagnostics, plus a 32-projection normal scalar-pair negative-control
    envelope. The normal null is never labeled F1.
  - `03-kernel-mechanism.png`: explanatory learned stem scalar-to-F1 kernel
    template and coefficient summary.
  - `04-spatial-latent-pca.png`: the Spec 0040 fixed25 PCA-RGB spatial maps and
    all-25 paired local-coherence companion.
  - `07-all25-latent-orbits.png` and matching JSON: a 5x5 population view in
    which each patch has adjacent normal and SO(2) PCA-orbit panels, using all
    360 inferred angles. This guards against
    presenting the predetermined patch-12 trace as representative by itself.
  - `rotation-orbits.html`: one offline advisor page with mathematical
    motivation, side-by-side scale-labeled slider plots, current metrics,
    archived all-25 table, F1 controls, and SVG charts rendered directly from
    precomputed arrays (no embedded PNG dashboard).
  - `manifest.json`: exact source/checkpoint/archive hashes, seeded-control
    state/projection hashes, settings, selected example/F1 copies, and explicit
    exploratory label.
- Each figure and the HTML carry: “Fixed validation 25; continuous-angle
  exploratory visualization; not sealed-test evaluation.”

## Measurement Contract

For patch `n`, angle `theta`, and final scalar posterior mean
`mu_theta in R^(16x32x32)`, display PCA points from
`vec(mu_theta[:,D]) in R^9856`, where D is the central radius-14 disk with
616 cells. PCA is fitted once to all 360 points of one model/patch orbit; it
is a coordinate system for a trace, not a model metric. Its origin is the
orbit mean, and the HTML reports the first-two-PC explained-variance fraction.
The models get separate PCA fits. The accepted local package uses predetermined
fixed-25 index `12`; the renderer can produce additional separately labeled
dense examples when the CPU budget permits. The scatter panels use equal data
units per pixel on both axes, with padded bounds, so a circular orbit cannot be
made elliptical by the rectangular figure layout. They show raw PCA-score tick
labels, but the two PCA bases are separate: numerical PC ranges cannot be used
as a cross-model magnitude comparison.

The all-25 quantity, evaluated at a documented sparse subset of the 1-degree
sweep when CPU-only inference makes the complete 25x360 grid impractical, is
the disk-masked relative RMS

`e_mu(theta) = ||mu(R_theta x) - R_theta mu(x)||_D /
                (||R_theta mu(x)||_D + eps)`.

The expected latent rotation uses the same interpolation convention at 32x32.
Report the median and interquartile envelope across patches. Do not call this a
formal equivariance-error result. The two archived `latent_mu.pt` inputs used
for this summary have recorded, enforced SHA-256 identities in the output
manifest; the renderer fails closed if either differs.

The dense all-25 extension is a distinct descriptive diagnostic. For each
model and patch, fit one two-component PCA over its 360 disk-masked raw
posterior means and use that PCA only for display. Preserve isotropic PCA axes
and close the displayed line with an exact copy of angle 0. Quantify geometry
in the disk-masked raw posterior rather than in PCA coordinates:

- cyclic local-linearity ratio = RMS(second one-degree difference) /
  RMS(first one-degree difference), where lower is locally smoother at this
  sampling step;
- coefficient of variation of the cyclic one-degree step RMS, where lower is
  more uniform in traversal speed;
- first-two-PC explained variance, where higher means more planar but does not
  establish equivariance.

Report paired favorable counts and medians across the fixed 25 without a
bootstrap, significance claim, checkpoint selection, or sealed-test claim.
Do not infer a general SO(2) advantage unless the realized population evidence
supports it consistently. A coarser 10-degree exploratory render is invalid as
final evidence because visible aliasing can change the apparent orbit shape.

The HTML must define

`RMS_D(A) = sqrt((1 / (16 |D|)) sum_{c=1}^{16} sum_{p in D} A_c(p)^2)`

before defining `e_mu`. It must state how to read it: `0` is exact agreement,
`1` means the RMS mismatch equals the expected field's RMS magnitude, and
smaller is better for this narrow proxy. Raw posterior RMS is contextual scale,
not a second score; it makes a near-zero denominator visible. The HTML must
also state the observed all-25 exact-quarter result: the normal medians
`1.327492/1.303358/1.326518` are lower than the SO(2) medians
`1.472552/1.417673/1.486783` at 90/180/270 degrees. Therefore this artifact
has mixed evidence and cannot claim a final-posterior SO(2) advantage.

For the slider's scale check, report the raw disk RMS of `mu_theta` over all 16
channels beside the selected-angle residual. It uses the common final-posterior
coordinate system and prevents a visually compact PCA trace from being mistaken
for a large latent magnitude. It is contextual magnitude, not an equivariance
score.

For the SO(2) internal-field panel, capture the final encoder D feature before
`mu_head`. Its packed shape is `B x 144 x 32 x 32 = 48 F0 + 48 F1x2`, while
the final `mu` remains 16 scalar channels. For one preregistered F1 copy `j`,
with two-vector `h_j(p)` at spatial coordinate `p`, use the uniform mean on
the same centered latent disk:

`bar_h_j(theta) = |D|^-1 sum_(p in D) h_j,theta(p)`.

Select the top three copies by their median initial vector magnitude across the
25 patches; exclude near-zero individual arrows using a recorded fixed
threshold. After initial-phase normalization, the expected trace is
`exp(i theta)`. Report phase error
`wrap(arg(z_theta)-arg(z_0)-theta)`, gain `|z_theta|/(|z_0|+eps)`, and raw
amplitude separately. The normal VAE has no F1 representation. Its optional
negative-control null instead uses 32 predeclared seeded orthonormal
96-to-48x2 scalar pairings, selected independently under the same fixed-25
theta-zero rule and shown only as a 5th--95th-percentile envelope. Record every
projection seed and one aggregate projection SHA-256; do not call it normal F1.

The seeded untrained SO(2) control reuses the trained model's selected F1 copy
indices. It isolates the transformation imposed by the architecture and cannot
prove that training learned equivariance or recover the historical training
initialization. Record its seed, state hash, and PyTorch version.

The kernel panel is explanatory only: select the stem's scalar-to-F1 filter
with the largest L2 norm and show its learned vector-field template,
magnitude, and compact coefficient summary. It demonstrates the architecture's
steerable mechanism, not learned data-level equivariance and not a baseline
comparison; the jointly rotated scalar-to-F1 kernel is invariant by
construction, so it must not be presented as a learned “steering orbit.”

## HTML Contract

The local HTML explains each visualization's mathematical construction and
motivation, shows normal and SO(2) trajectories side by side, and has only a
`0..359` angle slider. It updates selected points, direct residuals, F1 phase
markers, gain, phase error, and raw magnitude from already precomputed arrays;
it runs no model. Every chart is browser-native local SVG rather than a small
embedded raster image, so opening one file needs neither a network connection
nor a local server. Do not fabricate a normal-VAE F1 phase trace.

The slider is placed immediately above the two PCA orbit panels, after their
construction explanation, so its selected angle is visibly associated with
those panels. Use browser-native MathML with embedded LaTeX annotations for
the PCA construction, the scalar-posterior residual, and the F1 diagnostic.
The F1 explanation must distinguish the 48 two-component `F1` copies from the
16 final scalar posterior channels; define disk pooling, complex
phase-normalization, phase error, gain, and raw amplitude. It must say exactly
what the 32 seeded normal scalar-pair constructions are—and that they are a
negative-control null, never normal `F1` or an equivariance baseline.
Because a local browser does not include a TeX engine, inline `\\(...\\)` prose
expressions are converted after template insertion to styled Unicode plus
subscript/superscript HTML. There is no external MathJax or network dependency.

## Acceptance Criteria

- The renderer rejects wrong checkpoint hashes, non-60,000-update payloads,
  mismatched fixed-25 originals, unexpected shapes, or absent F1 packing.
- Zero-degree latent residual is numerically near zero; the 360 display point
  equals the saved 0-degree point exactly.
- The F1 selection, support mask, rotation convention, PCA fit scope, source
  hashes, fixed control seeds/state/projection hashes, and every output path
  are recorded.
- All three PNGs open successfully, are nonempty, carry the exploratory label,
  and the HTML contains no remote URL, raster embedding, or model/checkpoint/
  data dependency.
- Every numeric plot exposes its axis scale; raw PCA axes are labeled as
  within-model coordinates, while residual and raw posterior-RMS values make
  cross-model scale differences inspectable.
- The static figures are readable on a laptop; no screenshot dashboard or
  animated export is required.
- The dense all-25 JSON contains exactly 360 angles and 25 rows per model,
  binds both accepted checkpoint hashes and all 25 selected-patch hashes, and
  records the GPU runtime. The canonical PNG preserves the full 5x5 source;
  Spec 0048 reflows it into readable report Figures 8a/8b. Neither output may
  reuse the rejected 10-degree exploratory result.

## Tests And Verification Commands

- Focused synthetic tests cover continuous rotation identity, interpolation
  geometry, disk masking, orbit closure, PCA fit scope and aspect preservation,
  F1 unpacking/selection, pinned-archive identity, kernel choice,
  output-manifest contract, and offline HTML construction.
- `./.venv/bin/python -m pytest -q tests/test_frozen_vae_rotation_orbits.py`
- `./.venv/bin/ruff check src/eqvae/artifacts/rotation_orbits.py src/eqvae/cli/render_frozen_vae_rotation_orbits.py tests/test_frozen_vae_rotation_orbits.py`
- `./.venv/bin/ruff format --check src/eqvae/artifacts/rotation_orbits.py src/eqvae/cli/render_frozen_vae_rotation_orbits.py tests/test_frozen_vae_rotation_orbits.py`
- `./.venv/bin/basedpyright src/eqvae/artifacts/rotation_orbits.py src/eqvae/cli/render_frozen_vae_rotation_orbits.py`
- `./scripts/kaggle_kernel.sh build kaggle/kernels/fixed25_rotation_population`
- `./scripts/kaggle_kernel.sh validate kaggle/kernels/fixed25_rotation_population`
- Run the renderer on the accepted local inputs, inspect all PNGs, then run
  `./scripts/python_quality.sh` and `git diff --check`.

## Known Risks And Adversarial Checks

- Interpolated rotation, PCA coordinates, and the kernel panel can overstate
  performance. The outputs must visibly separate exploratory visualization,
  formal exact-rotation evidence, and architecture mechanics.
- A square spatial mean is not rotation-invariant at its corners; reject it in
  favor of the recorded disk mean. Check that phase channels are neither summed
  across all 48 copies nor confused with the 16-channel scalar `mu`.
- Ensure a “circle” cannot be manufactured by fitting PCA independently at each
  angle, stretching one PCA plotting axis, or by calling an arbitrary normal
  scalar-pair null an F1 construction.

## Implementation Record

The dense `0..359` trajectory below is preserved only as historical provenance.
Spec 0050 proved that it spliced opposite rotation conventions at 90 and 270
degrees, so its smoothness, step-CV, PCA-shape, and continuous-angle pooled-F1
claims are invalid for scientific use. Its bytes and hashes remain immutable;
the separate exact-`torch.rot90` quarter-turn controls were not affected.

- The dense all-25 extension adds reusable population collection, raw-space
  summaries, and rendering to the existing artifact module and CLI, with a
  compact guarded Kaggle kernel at
  `kaggle/kernels/fixed25_rotation_population`. The one authorized private T4
  run completed as
  `maximshtefan/eqvae-fixed25-dense-rotation-population/1`; Kaggle normalized
  the requested slug and the launch receipt preserves both identities at
  `runs/local/kaggle_launches/maximshtefan/eqvae-fixed25-dense-rotation-population/v0001.json`.
- The superseded 1-degree output contains exactly 360 angles and 25 rows/model.
  SO(2) has the lower local-linearity ratio in 25/25 pairs, with median
  `0.983993` versus normal `1.033922`; it has the lower step-size CV in 0/25,
  with median `0.204670` versus `0.189132`; and higher two-PC explained
  variance in 9/25, with median `0.038098` versus `0.039123`. This supports a
  formerly supported a restricted claim of greater one-degree local smoothness;
  that claim is withdrawn by Spec 0050.
  GPU measurement time was `321.53 s` on a Tesla T4 with PyTorch
  `2.10.0+cu128`. Final PNG/JSON SHA-256 values are
  `bc29ee7e1f4e42177e4cb6d37da0571014d476736e31f153bc88e0f8cef853ac`
  and `f80be0451d0d77b3dfb906100cb4576615939f4b19fb78a95ec6e823f5901066`.

- Implemented in `src/eqvae/artifacts/rotation_orbits.py` and
  `src/eqvae/cli/render_frozen_vae_rotation_orbits.py`, with focused coverage
  in `tests/test_frozen_vae_rotation_orbits.py`.
- The accepted local render used dense angles `0..359` for predetermined fixed
  patch 12, canonical `docs/data/fixed25/originals.pt`, and the hash-checked
  frozen checkpoints. The two archived all-25 latent sources are SHA-pinned and
  their verified hashes are written into the manifest. Outputs are in ignored
  `runs/local/frozen_vae_rotation_orbits/`.
- The PCA bounds enforce an equal data-unit-to-pixel scale on both axes;
  focused coverage prevents a circular input trace from becoming an ellipse
  merely because the panel is rectangular. Static plots show numeric ticks;
  the advisor page places both models side by side, states the one-fit
  9,856-dimensional PCA construction and explained variance, and renders all
  of its own charts from local SVG data rather than embedded PNGs.
- The posterior-orbit comparison gives exploratory trace geometry only. This
  does **not** agree with the archived masked all-25 exact-quarter residual ranking:
  normal medians are `1.327492/1.303358/1.326518` at 90/180/270 degrees versus
  SO(2) `1.472552/1.417673/1.486783`. Therefore present Figure 01 only as
  exploratory geometry with mixed evidence, never as an SO(2) performance win.
- The predeclared spatially pooled F1 diagnostic did **not** form the expected
  clean phase circle for this checkpoint, including under a circular-aperture
  check. Figure 02 now separates phase error, gain, and raw amplitude; it
  contrasts trained F1 copies with the same copies in a seeded untrained SO(2)
  architecture control and a 32-projection normal scalar-pair negative-control
  envelope. These are construction/null diagnostics, not positive evidence or
  model-quality baselines.
- The full v3 local render completed with dense angles `0..359`, schema v3
  manifest, control source identities, parse-checked offline JavaScript, and
  inspected 1800x1120/2400x1520 PNGs. Focused tests (10), touched-file
  Ruff/format and BasedPyright pass. The full
  `./scripts/python_quality.sh` gate currently fails before tests on unrelated
  existing lint failures in `kaggle/kernels/tissue_fastpath_probe/run_template.py`
  and `kaggle/kernels/wsi45630_full_compile_probe/package/kernel/run.py`; this
  workstream does not modify those files.
- The explanation-layout refresh uses browser-native MathML plus embedded LaTeX
  annotations to derive PCA, relative RMS, and the F1 diagnostic. It moves the
  angle slider immediately above the PCA panels and explicitly states that the
  normal VAE is lower on the archived exact-quarter residual proxy. Its
  numerical payload is unchanged: the existing dense-360 JSON was re-emitted
  into the local HTML, and a separate angles-0--3 renderer smoke pass, focused
  tests (10), touched-file Ruff/format, BasedPyright, diff check, and
  JavaScript parse check passed.
- Spec 0040 added the fourth spatial-PCA figure and the advisor page's Section
  5 from the SHA-pinned archived fixed25 posterior tensors, with no model
  rerun. The resulting normal/SO(2) median relative edge RMS is
  `1.456673/1.467611`; the paired SO(2)-normal mean is `+0.010646` (23/25
  positive), so the implementation explicitly makes no SO(2) local-coherence
  claim. The manifest schema is now v4. Focused tests (12), archive validation,
  JS parse, and visual PNG inspection pass.

## Related Files

- `docs/specs/0010-fixed25-equivariance-artifact-protocol.md`
- `docs/decisions/0009-fixed25-embedding-equivariance-eval-proxy.md`
- `src/eqvae/artifacts/fixed25_equivariance.py`
- `src/eqvae/inference/checkpoints.py`
- `src/eqvae/models/so2_vae.py`
- `src/eqvae/models/so2_architecture_probe.py`
