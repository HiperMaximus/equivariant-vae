# MIDL 2027 manuscript

Format-only adaptation applied in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24)
on 2026-10-08, then imported from its source download (`main.tex` v147,
including the subsequent author-email update).
The baseline was accepted INCISCOS `main.tex` v145, published in research
commit `7317813c2151d5c2d8f79d57690aeb068a95365d`.
The original `../inciscos2026` manuscript remains available separately.
The format edits remain tracked suggestions in Overleaf. All 11 professor
comment threads retain their identities, messages and resolution states
(five open, six resolved); the existing `main.tex` document was edited in place.

The scientific title, abstract, keywords, section order, prose, equations,
citations, captions, table cells, bibliography and figure sources are retained.
The new document uses the official MIDL full-paper class, its default
single-column layout, letter paper, font sizes, line spacing, margins and
author-year citations. Tables use the class's normal text size; the MIL
diagram and two table layouts were adjusted to fit the text width.
No page-limit cuts have been applied. The current PDF has 19 body pages and
three reference pages (22 total).

Authors are Maximiliano Garavito Chtefan and David Edmundo Romo Bucheli, in
that order, with `hipermaximus@gmail.com` and `deromob@saber.uis.edu.co`,
respectively. The affiliation follows the earlier SIPAIM scaffold.
The MIDL class is used without its anonymization option.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The output is `midl2027.pdf`. The tracked PDF was downloaded from Overleaf
after compilation; its SHA-256 is
`724b027720cbecc3d917614206e61549177c27266465ac2ceeea83a1e4ebdfd9`.
All 28 source, bibliography, figure and class files match the Overleaf source
download byte for byte. Only the three new class files were uploaded;
the manuscript text was edited in place. Continue editing in Overleaf first,
then import its current sources and compiled PDF into this directory.
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

The complete scientific body matches the INCISCOS source after normalizing
only the documented formatting commands. Bibliography and all copied figure
files are byte-identical to their INCISCOS counterparts. The original
INCISCOS source and PDF are unchanged. The final PDF compiles without errors,
undefined references or overfull boxes. All 22 rendered pages were inspected;
The Overleaf build reports four nonfatal underfull-box notices.
BibTeX also reports that the existing `im_denoising_2017` entry has both
`volume` and `number`; the default `plainnat` style uses the volume. The
bibliography database was retained unchanged for this format-only adaptation.
