# MIDL 2027 manuscript

The active manuscript is maintained in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24)
and was imported on 2026-10-09 from live `main.tex` v453 and
`references.bib` v4. This includes the professor's edits and the user-approved
abstract, introduction, contribution statement, quantitative related-work
context and Lafarge reference. Both author contact details match Overleaf.
All fourteen comment threads retain their histories, anchors and resolution
states (eight open, six resolved), including the three approved replies.
Tracked suggestions remain in Overleaf; none were accepted or rejected during
the read-only import, and no comment was closed.

The initial MIDL format adaptation on 2026-10-08 used accepted INCISCOS
`main.tex` v145, published in research commit
`7317813c2151d5c2d8f79d57690aeb068a95365d`, as its baseline.
The original `../inciscos2026` manuscript remains available separately.
The current document uses the official MIDL full-paper class, its default
single-column layout, letter paper, font sizes, line spacing, margins and
author-year citations. Tables use the class's normal text size; the MIL
diagram and two table layouts were adjusted to fit the text width.
No page-limit cuts have been applied. The current PDF has 19 body pages and
three reference pages (22 total).

Authors are Maximiliano Garavito Chtefan and David Romo-Bucheli, in
that order, with `maximiliano2162094@correo.uis.edu.co` and `deromob@uis.edu.co`,
respectively. The affiliation follows the earlier SIPAIM scaffold.
The MIDL class is used without its anonymization option.

The abstract anticipates the planned evaluation on 152 WSIs excluded from
VAE learning. Its `XXXX`/`YYYY` values and provisional significance statement
remain pending new results, with the approved TODO in the source. The body
still reports the historical 23-WSI classification evaluation. This is an
intermediate manuscript, not a completed report of the new experiment.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The output is `midl2027.pdf`. The tracked PDF was compiled locally from the
current Overleaf sources on 2026-10-09; its SHA-256 is
`1e908e8dd138ab4988067a107152c57d92f0f6e66bccefce48ee64e2651e1817`.
All 28 active source, bibliography, figure and class files match live Overleaf
byte for byte. Only `main.tex` and `references.bib` differed from the earlier
local import. Continue editing in Overleaf first, then import all current
changes and regenerate the matching PDF. The user's standing authorization
requires committing and pushing the paper updates to research `origin/main`
before declaring a correction round complete, verifying source parity and the
published PDF. Preserve unrelated research work and professor comment states.
The remaining files in the Overleaf tree are historical IEEE build assets and
an unused uploaded `inciscos2026.pdf`; that uploaded PDF is not the current
MIDL build. The archived INCISCOS repository copy is preserved separately.
The bundled classes require standard TeX Live packages, including `natbib`,
`algorithm2e` and TikZ.

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

## Verification

The imported manuscript matches the approved live Overleaf revision; figures
and class assets are unchanged. The original INCISCOS source and PDF are
unchanged. The current PDF compiles without errors, undefined references or
overfull boxes. All 22 rendered pages were inspected. Its first page contains
the institutional author email and no personal email. The build reports six
nonfatal underfull-box notices.
BibTeX also reports that the existing `im_denoising_2017` entry has both
`volume` and `number`; the default `plainnat` style uses the volume. The newly
added Lafarge citation appears in the introduction and bibliography.
