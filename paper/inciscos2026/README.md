# INCISCOS 2026 paper

This directory is the anonymous INCISCOS 2026 adaptation of the earlier SIPAIM
paper scaffold. It uses the standard IEEE conference A4 format requested by
INCISCOS; the conference does not provide a separate LaTeX class.

Submission constraints checked on 2026-09-16:

- English;
- IEEE conference format on A4 paper;
- 4--8 pages;
- double-blind review, with no author or affiliation details;
- PDF or Microsoft Word submission.

Build the paper from this directory:

```bash
latexmk -pdf main.tex
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
