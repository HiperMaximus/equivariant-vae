# MIDL 2027 manuscript

The active manuscript is maintained in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24),
which is the source of truth. The current import is the complete project source
export reviewed on2026-10-28 after the approved MIL representation equation. All28 active
sources/assets match that export byte-for-byte; only main.tex changed in
this correction round. All own tracked suggestions are accepted.
All22 professor comment histories are retained, with5 open and17 resolved.
Only the professor closes comments.

## Current related-work revision

Section2.1, Unsupervised Representation Learning for Histopathology, covers
Tellez and Lafarge. Section2.2, Equivariance for Deep Learning, explains the
transformation law, equivariance versus invariance, weight sharing and
conditional data-efficiency benefits, and group/steerable implementations.
TARGET-VAE and O2-VAE now appear in this general section, with their imaging
domains stated. O2-VAE specifies 2D segmentation masks and grayscale images
from fluorescence microscopy, including confocal data, alongside synthetic
cell-shape/texture data.

Each prose paragraph occupies one physical source line. Paragraph separators,
equations, tables, TikZ and deliberate LaTeX line breaks remain structural.
The individually approved CNN, TARGET-VAE, O2-VAE and general-equivariance
replies are posted. The old kernel-support reply was replaced in place with
the approved update. All five open comments have been reviewed; their states
remain open. Only the professor resolves comments.

The Materials and Methods introduction is followed by task-specific partitions.
MIL reuses the361 VAE-development WSI (322 VAE train +39 VAE validation) in
five diagnosis-stratified WSI folds:288/73 train/validation for fold0 and289/72
for folds1--4. The final classifier is initialized afresh and trained on all361
WSI with a cross-validation-derived fixed duration, no validation split and no
validation-based early stopping. The separate152-WSI cohort supplies final
MIL evaluation. The manuscript describes this protocol in the present tense;
final refitting and evaluation remain experimentally pending.

The abstract, population table, experimental design, MIL methodology, statistics,
results placeholders, discussion, limitations and conclusion reflect that
protocol. Figure4 now separates the reused VAE/MIL development cohort from the
152-WSI MIL evaluation cohort and the106/23/23 tissue partition. Reconstruction
still uses the23 tissue-test WSI. Figure10 retains the accepted tissue curve as
a vector plot; the incompatible23-WSI MIL panel and numerical summaries are
replaced by pending152-WSI placeholders. The unchanged raster asset remains
preserved in the project. No previous-experiment narrative is added to the
manuscript.

The professor's review marker and152/361 formatting are preserved. The five
comment threads remain open; no replies were posted. Two threads inadvertently
marked resolved during own-change acceptance were immediately reopened to
restore their original states; their histories remain intact.

## Build and verification

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The24-page PDF was rebuilt from the imported sources/assets. All pages were
rendered and visually checked. The approved patch-to-encoder explanation,
diagonal Gaussian posterior and deterministic mean representation appear in
Equation1 on page7, followed by the definition of the MIL bag. The prose omits
"large" and refers back to the existing foreground-selection description.
Each prose paragraph remains on one physical source line. Compilation has no
errors, undefined citations or overfull boxes. PDF SHA256:
`491c26c197e85334d3ffdc6710d968f6e9bd5520c414597d43ca9164ba53d230`.

The official MIDL full-paper class and its default single-column layout,
letter paper, typography, margins and author-year citations are preserved.
Authors are Maximiliano Garavito Chtefan and David Romo-Bucheli. The template
is used without anonymization; no page-limit cuts have been applied.

The abstract and WSI result tables retain XXXX/YYYY placeholders for the
pending152-WSI evaluation. The final MIL recipe and numerical duration still
require completion from cross-validation. Reconstruction and tissue results
remain unchanged.

After Overleaf changes, import all active sources/assets and build the matching
PDF, preserving professor edits and comments. Publish only paper and relevant
handoff/policy changes to research origin/main. Never send the research repo
to Overleaf. Historical INCISCOS/SIPAIM copies remain separate.

The current review uses the live source export and
`/tmp/paper-mil-protocol-20261026/` for temporary comparison and rendering.
Earlier related-work evidence is in
`../../../.agent-tools/overleaf-review/review-data/midl-final-style-2026-10-16/`.
Previous accepted imports and publication receipts are recorded in CURRENT.md.

## Official template and dependencies

- [MIDL 2027 author instructions](https://2027.midl.io/author-instructions)
- [Official MIDL template](https://github.com/MIDL-Conference/MIDLLatexTemplate),
  revision `f783cacb7146990d4cf22c87af64445474d83acd`;
  `midl.cls` is copied without modification.
- [JMLR/PMLR class bundle](https://ctan.org/pkg/jmlr), version 1.30;
  `jmlr.cls` and `jmlrutils.sty` were generated from the official CTAN
  `jmlr.dtx` with `jmlr.ins` and copied without modification.

MIDL and the JMLR bundle are distributed under the LaTeX Project Public
License; the bundled files retain their original license notices.
