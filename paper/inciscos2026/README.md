# INCISCOS 2026 paper

The former SIPAIM Overleaf project now contains a curated copy of this
manuscript at https://www.overleaf.com/project/69c614433cbc9e46cf226d24.
The export includes `main.tex`, `references.bib`, the manuscript figures,
the bundled IEEE class/style, `.latexmkrc`, and `inciscos2026.pdf`. It excludes
the personal EPUB, rendering scripts and issue-preview images. The local
`paper/sipaim2026` scaffold remains available separately.

This directory is the anonymous INCISCOS 2026 adaptation of the earlier SIPAIM
paper scaffold. It uses the standard IEEE conference A4 format requested by
INCISCOS; the conference does not provide a separate LaTeX class.

The current build is the complete professor-review draft; no venue-length cuts
have been applied. It is organized as a direct experimental report of the
conventional and continuous-`SO(2)` VAEs, reconstruction, tissue
recognition, and WSI diagnosis. Later geometry experiments are deliberately excluded.
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
MPLCONFIGDIR=/tmp/mplconfig ../../.venv/bin/python make_figures.py
```

The figure script requires NumPy, Matplotlib, PyTorch, and Pillow. It does not
recompute metrics or access sealed labels. It renders accepted, hash-bound
results and the first nine fixed validation reconstructions in selector order.
The reconstruction boxplots describe the 23 accepted per-WSI mean metrics;
they are not confidence intervals. The script also creates
`figures/vae_training_curves.png` as a separate review preview. Its shaded
validation band is the recorded mean ± one standard deviation within each
validation evaluation, not variation across independent training runs. This
preview is not included in the manuscript.

The bundled `IEEEtran.cls` and `IEEEtran.bst` are the same standard IEEE files
used by the earlier paper subtree.

The review PDF is `inciscos2026.pdf`. Restore author names, affiliations and
acknowledgments only after acceptance.


## Personal Kindle EPUB

The local Pandoc 3.1.3 installation lives in the ignored
`../../.venv/pandoc-local` directory. From the repository root, after
building the PDF, run:

```bash
.venv/bin/python paper/inciscos2026/make_epub.py
```

The output is `inciscos2026_kindle.epub`. The script keeps text
reflowable, converts equations to MathML and embeds the five LaTeX diagrams as
images cropped from the compiled PDF. Its crop coordinates match this
ten-page review layout; update them if pagination changes. This EPUB is a
personal reading copy, not the conference submission file.

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

The mask panels in Figure 1 use the official UBC-OCEAN `train_thumbnails/10143_thumbnail.png`
and the matching `10143.png` from the public supplemental-mask dataset. Both
correspond to a training-cohort WSI of `40063 × 42207` pixels. The original
thumbnail (`3000 × 3160`) was resized to `600 × 632` and saved as JPEG
(quality 75). The full-resolution mask was streamed through
`pngtopnm -byrow`, `pnmscale -xysize 1200 1264`, and `pamtopng` to avoid
loading the entire image into memory, then reduced to `600 × 632` by
nearest-neighbor resampling. The overlay was reduced to the same size and saved
as JPEG (quality 75); it retains the thumbnail beneath the painted mask
regions. The mask is partial; black pixels are unannotated.

The TMA panel in Figure 1 compares that WSI with the official UBC-OCEAN `train_images/13568.png`,
a training TMA of `2964 × 2964` pixels. Its full-resolution source was resized
to `600 × 600` and saved as JPEG (quality 75) for display. The panels are
shown at similar heights, not a shared physical scale. The TMA source SHA-256 is
`3d0c3b61623d8a01dc0f858faa7ed896a56ff111c252b5037e7e10f790f7fbe0`.

Illustrative thumbnails use reduced JPEG rasters to keep PDF navigation responsive.
The diagnostic patch crops and quantitative result figures retain their original
resolution and lossless format.

The residual-block figure shows the same-resolution building block of each VAE,
including its two convolutions, normalization, gate, identity skip and
post-addition gate. The caption describes the branch-local stage transitions.
The architecture figure is drawn from the frozen model implementations in
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
in the earlier eleven-page review PDF linked in issue #7. They include the
figure captions, were rendered at 180 dpi, and are 1288 pixels wide. Histology and reconstruction previews
use optimized JPEG (quality 82); diagrams and plots use optimized PNG. The nine
files together occupy about 1.24 MB. The current PDF and LaTeX sources remain the manuscript source of truth;
these crops document the earlier GitHub issue comment.

## MIL concept figure provenance

The MIL illustration uses the independently recorded UBC-OCEAN thumbnail
for WSI `11557`, already reduced to `640 x 541` pixels in the thesis assets.
Its source PNG SHA-256 is
`43f5294cd3b019f88a4b154215317c7fab9e8df8b7a6357309dd1fda43be1e00`.
`mil_11557_thumbnail.jpg` is a quality-82 JPEG export. The six quality-88
JPEG regions are `48 x 48` thumbnail pixels, centered at `(330,110)`,
`(150,150)`, `(480,200)`, `(340,300)`, `(130,320)`, and `(490,380)`.
These are illustrative overview regions, not the full-resolution model
patches. No new WSI download, inference or metric calculation was performed.
All seven added JPEGs total 106,825 bytes.
