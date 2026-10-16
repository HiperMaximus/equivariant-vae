# MIDL 2027 manuscript

The active manuscript is maintained in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24),
which is the source of truth. The current import is main.texv954 and
references.bibv5 (2026-10-16). All28 active sources/assets match Overleaf.
All tracked edits authored by Maximiliano are accepted. All19 comment
histories and states are preserved:7 open and12 resolved.

## Current related-work revision

Section2.1, Unsupervised Representation Learning for Histopathology, covers
Tellez and Lafarge. Section2.2, Equivariance for Deep Learning, explains the
transformation law, equivariance versus invariance, weight sharing and
conditional data-efficiency benefits, and group/steerable implementations.
TARGET-VAE and O2-VAE now appear in this general section, with their imaging
domains stated. O2-VAE specifies 2D segmentation masks and greyscale images
from fluorescence microscopy, including confocal data, alongside synthetic
cell-shape/texture data. MIL and the existing figures are preserved.

Each prose paragraph occupies one physical source line. Paragraph separators,
equations, tables, TikZ and deliberate LaTeX line breaks remain structural.
The three related-work replies await individual human approval; the approved
CNN reply has already been posted. Only the professor resolves comments.

## Build and verification

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The22-page PDF was rebuilt from sources/assets byte-identical to the import.
All pages were rendered and visually reviewed. Compilation has no errors,
undefined citations or overfull boxes. BibTeX retains only the existing
volume/number warning in im_denoising_2017. PDF SHA256:
`6bf02edad4f508adda3c1695d25765f78984cbf0fd4a20cec799507a72ced613`.

The official MIDL full-paper class and its default single-column layout,
letter paper, typography, margins and author-year citations are preserved.
Authors are Maximiliano Garavito Chtefan and David Romo-Bucheli. The template
is used without anonymization; no page-limit cuts have been applied.

The abstract still anticipates evaluation on152 WSIs with XXXX/YYYY and its
existing TODO; the body retains the historical23-WSI evaluation. These are
pending scientific results, not compilation failures.

After Overleaf changes, import all active sources/assets and build the matching
PDF, preserving professor edits and comments. Publish only paper and relevant
handoff/policy changes to research origin/main. Never send the research repo
to Overleaf. Historical INCISCOS/SIPAIM copies remain separate.

Evidence for this revision is in
`../../../.agent-tools/overleaf-review/review-data/midl-related-work-2026-10-16/`.
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
