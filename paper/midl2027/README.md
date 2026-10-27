# MIDL 2027 manuscript

The active manuscript is maintained in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24),
which is the source of truth. The current import is the live main.tex text
captured on 2026-10-26 after accepting the individually approved Materials
and Methods introductory paragraph. The other27 active sources/assets match
the project source export reviewed that day, including references.bibv5.
All tracked edits authored by Maximiliano are accepted. All22 comment
histories and states are preserved:5 open and17 resolved; the professor
resolved five previous threads on2026-10-23.

## Current related-work revision

Section2.1, Unsupervised Representation Learning for Histopathology, covers
Tellez and Lafarge. Section2.2, Equivariance for Deep Learning, explains the
transformation law, equivariance versus invariance, weight sharing and
conditional data-efficiency benefits, and group/steerable implementations.
TARGET-VAE and O2-VAE now appear in this general section, with their imaging
domains stated. O2-VAE specifies 2D segmentation masks and grayscale images
from fluorescence microscopy, including confocal data, alongside synthetic
cell-shape/texture data. MIL and the existing figures are preserved.

Each prose paragraph occupies one physical source line. Paragraph separators,
equations, tables, TikZ and deliberate LaTeX line breaks remain structural.
The individually approved CNN, TARGET-VAE, O2-VAE and general-equivariance
replies are posted. The old kernel-support reply was replaced in place with
the approved update. All five open comments have been reviewed; their states
remain open. Only the professor resolves comments.

The Materials and Methods section now introduces the dataset, patch selection,
WSI partitions, matched VAEs and the three evaluation tasks. The professor's
latest review marker and mathematical formatting of152/361 are preserved.
The partition update, MIL-bag wording, split figure and class-order explanation
remain pending individual review. No comment replies accompanied this change.

## Build and verification

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The23-page PDF was rebuilt from the imported sources/assets.
All pages were rendered and visually checked, including the introductory
paragraph on page4. Compilation has no errors,
undefined citations or overfull boxes. BibTeX retains only the existing
volume/number warning in im_denoising_2017. PDF SHA256:
`0124a3d254493c0a437829543bc5d51b8b4af4d0fd0a8bf3e3cbbd02574cdc35`.

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
