# INCISCOS 2026 paper

This directory is the anonymous INCISCOS 2026 adaptation of the earlier SIPAIM
paper scaffold. It uses the standard IEEE conference A4 format requested by
INCISCOS; the conference does not provide a separate LaTeX class.

The current build is the complete professor-review draft; no venue-length cuts
have been applied. It is organized as a direct experimental report of the
conventional and continuous-`SO(2)` VAEs, reconstruction, WSI diagnosis, and
tissue recognition. Later geometry experiments are deliberately excluded.
Content can be shortened for submission after academic review.

Submission constraints checked on 2026-09-16:

- English;
- IEEE conference format on A4 paper;
- 4--8 pages;
- double-blind review, with no author or affiliation details;
- PDF or Microsoft Word submission.

Build the tracked review PDF from this directory:

```bash
latexmk -pdf -jobname=inciscos2026 main.tex
```

Regenerate the English-language result figures from the accepted local evidence:

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=/tmp/inciscos-fig-deps \
  python make_figures.py
```

The figure script does not recompute metrics or access sealed labels. It only
renders values already present in the accepted, hash-bound result artifacts.
The bundled `IEEEtran.cls` and `IEEEtran.bst` are the same standard IEEE files
used by the earlier paper subtree.

The review PDF is `inciscos2026.pdf`. Restore author names, affiliations and
acknowledgments only after acceptance.

## Dataset figure provenance

Figure 1 uses the recorded full-resolution crops at `(13056, 4608)` and
`(44544, 9984)` from VAE-validation WSI `59031`. The official UBC-OCEAN
`train_thumbnails/59031_thumbnail.png` is `3000 × 2488` pixels and was
reduced to `800 × 663` and saved as JPEG (quality 75) for the figure. Its
aspect ratio differs from the
`48866 × 47340` dimensions in the public `train.csv`, so projecting the crop
coordinates using that CSV puts the markers in the wrong locations. Matching
each actual crop after a 16 × 16 reduction to the official thumbnail locates
the crop top-lefts at `(858, 303)` and `(2925, 656)`; their displayed center
positions are `(866, 311)` and `(2933, 664)` on the official thumbnail. Both
matches have about 50 RGB squared-error units per channel, versus about 198
and 336 at the naive CSV projection. The arrows use these matched centers.
The thumbnail is an illustration and was not a model input. The official
thumbnail SHA-256 is
`069d98a56c6fb4269f7b53c2e62e1c86bb0c1756d35e836d72faed87b5f314b7`.

Figure 2 uses the official UBC-OCEAN `train_thumbnails/10143_thumbnail.png`
and the matching `10143.png` from the public supplemental-mask dataset. Both
correspond to a training-cohort WSI of `40063 × 42207` pixels. The original
thumbnail (`3000 × 3160`) was resized to `600 × 632` and saved as JPEG
(quality 75). The full-resolution mask was streamed through
`pngtopnm -byrow`, `pnmscale -xysize 1200 1264`, and `pamtopng` to avoid
loading the entire image into memory, then reduced to `600 × 632` by
nearest-neighbor resampling. The overlay was reduced to the same size and saved
as JPEG (quality 75); it retains the thumbnail beneath the painted mask
regions. The mask is partial; black pixels are unannotated.

Figure 3 compares that WSI with the official UBC-OCEAN `train_images/13568.png`,
a training TMA of `2964 × 2964` pixels. Its full-resolution source was resized
to `600 × 600` and saved as JPEG (quality 75) for display. The panels are
shown at similar heights, not a shared physical scale. The TMA source SHA-256 is
`3d0c3b61623d8a01dc0f858faa7ed896a56ff111c252b5037e7e10f790f7fbe0`.

Illustrative thumbnails use reduced JPEG rasters to keep PDF navigation responsive.
The diagnostic patch crops and quantitative result figures retain their original
resolution and lossless format.

Figure 6 shows the same-resolution residual building block of each VAE,
including its two convolutions, normalization, gate, identity skip and
post-addition gate. The caption describes the branch-local stage transitions.
Figure 7 is drawn from the frozen model implementations in
`src/eqvae/models/non_equivariant_vae.py` and `src/eqvae/models/so2_vae.py`,
with the continuous-field mechanics in `so2_architecture_probe.py` and the
locked layer map in Spec 0014. Each diagram row contains the stem, four
encoder or decoder residual stages, the separate posterior heads, and the
latent projection or output head.

Source SHA-256: `train.csv`
`21daca35faf6e1c934875ebb1bfa2a6d9fabe783d6eb721884379bfe5e058266`;
`10143_thumbnail.png`
`e673345e36b2c7724f7f3dc91c7aef7862a52fc680128171a3733863691`;
`10143.png`
`19f7244ae8379a1b7a5aaeb3353796906bdff3c8bc42c151575db7bd74b16328`.

## Issue #7 figure previews

`issue_figures/fig01` through `fig09` are lightweight crops of the nine figures
in the eleven-page review PDF. They include the figure captions, were rendered
at 180 dpi, and are 1288 pixels wide. Histology and reconstruction previews
use optimized JPEG (quality 82); diagrams and plots use optimized PNG. The nine
files together occupy about 1.24 MB. The PDF and LaTeX sources remain the
manuscript source of truth; these crops are for the GitHub issue comment.
