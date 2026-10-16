# MIDL 2027 manuscript

The active manuscript is maintained in the existing
[Overleaf project](https://www.overleaf.com/project/69c614433cbc9e46cf226d24).
The current local copy imports the professor's completed 2026-10-16 edits
from `main.tex` v950, followed by the requested paragraph formatting in v951;
`references.bib` remains v4. All 28 active sources/assets match live Overleaf.
The nineteen comment histories and states are preserved: seven open and twelve
resolved. The professor resolved the tissue-budget abstract comment. Four new
open threads concern the original CNN reference, the image domains used by
TARGET-VAE and O2-VAE, and an equivariance-for-deep-learning subsection.

Each prose paragraph, list item and caption now occupies one physical source
line. Paragraph separators, words, formulas, tables, figures, comments and
intentional LaTeX line breaks were preserved. The 393 internal line breaks
were replaced with spaces through one tracked edit; all 783 tracked operations
were then accepted through Overleaf's UI. No suggestions remain pending.
Future manuscript edits follow this source-formatting preference and accept
our own tracked edits after each round, preserving professor suggestions and
comment states. See `../../CURRENT.md` and `../../AGENTS.md`.

The current document uses the official MIDL full-paper class and its default
single-column layout, letter paper, font sizes, line spacing, margins and
author-year citations. No page-limit cuts have been applied. The PDF has
19 body pages and three reference pages (22 total). Authors are Maximiliano
Garavito Chtefan and David Romo-Bucheli, in that order, with their institutional
emails. The MIDL class is used without its anonymization option.

The abstract anticipates the planned evaluation on 152 WSIs excluded from
VAE learning. Its `XXXX`/`YYYY` values and provisional significance statement
remain pending new results, with the existing TODO preserved. The body still
reports the historical 23-WSI classification evaluation.

## Build and verification

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=midl2027 main.tex
```

The output is `midl2027.pdf`, rebuilt on 2026-10-16 with SHA-256
`63a64dda51cfb09064f244e32efe49ae14794f29c9855ab1cab7a803a2746bd0`.
All 22 rendered pages were inspected. Both extracted text and every rendered
page are identical before and after the whitespace-only paragraph formatting.
Compilation has no errors or overfull boxes. BibTeX retains a warning for the
professor's empty `\cite{}` placeholder and the existing volume/number warning
in `im_denoising_2017`. The empty citation, ambiguous macro-F1 value ordering
in the abstract and wording about supervised learning remain review items;
the professor's text was preserved.

Live Overleaf remains the source of truth. After manuscript changes, import
all current sources/assets, regenerate and inspect the matching PDF, commit
only paper and relevant handoff/policy updates, and push to research
`origin/main` under the user's standing authorization. Never push the research
repository to Overleaf. The legacy IEEE assets and unused uploaded
`inciscos2026.pdf` are historical, and the separate INCISCOS repository copy is
preserved. Import, paragraph-formatting, comment and PDF verification evidence
is in `../../../.agent-tools/overleaf-review/review-data/` under
`midl-final-review-2026-10-16` and `midl-final-sync-2026-10-16`.

## Earlier imports

The 2026-10-09 import used `main.tex` v453 and `references.bib` v4, with fourteen
comment threads (eight open and six resolved). On 2026-10-13 all 195 pending
change ranges authored by Maximiliano were accepted, preserving fifteen
threads (nine open and six resolved). The initial MIDL adaptation used accepted
INCISCOS `main.tex` v145 and research commit
`7317813c2151d5c2d8f79d57690aeb068a95365d`; the INCISCOS manuscript remains
separate. The prior PDF hash and detailed historical handoff remain in
`../../CURRENT.md`.

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
