# SIPAIM 2026 Full Paper Template

This folder contains the LaTeX template base for a SIPAIM 2026 full paper.
SIPAIM 2026 points authors to the IEEE Manuscript Template for Conference
Proceedings; there is no separate SIPAIM-specific LaTeX class in this repo.

Official submission constraints checked on 2026-06-05:

- Event: 22nd International Symposium on Medical Information Processing and
  Analysis, November 18-20, 2026, Lima, Peru.
- Full paper deadline: July 9, 2026.
- Full paper length: maximum 4 double-column pages.
- Paper format: English, PDF, IEEE Manuscript Template for Conference
  Proceedings.
- Review mode: single-blind, so authors and affiliations are included.
- Accepted full papers are submitted for inclusion in IEEE Xplore, subject to
  IEEE scope and quality requirements.

Primary sources:

- SIPAIM submission page: https://sipaim.org/submission
- IEEE conference template page:
  https://ieee-bf.org/manuscript-templates-for-conference-proceedings/
- CTAN IEEEtran package: https://ctan.org/pkg/ieeetran
- IEEE official Overleaf gallery:
  https://www.overleaf.com/gallery/tagged/ieee-official

Project links:

- Current Overleaf project:
  https://www.overleaf.com/project/69c614433cbc9e46cf226d24
- Local repo source:
  `paper/sipaim2026`
- Overleaf sync workflow:
  `docs/overleaf_sync_workflow.md`

PDF policy:

- Keep `sipaim2026.pdf` updated and tracked for advisor/GitHub visibility.
- Treat `main.pdf` as local build output; it is ignored by git.

Template files:

- `template/IEEEtran/` contains the downloaded CTAN `IEEEtran` package.
- `template/IEEEtran/bare_conf.tex` is the bare IEEE conference starter file.
- `template/IEEEtran/IEEEtran.cls` is the IEEEtran class file.
- `template/IEEEtran/bibtex/IEEEtran.bst` is the bibliography style.
- `main.tex` is a minimal SIPAIM full-paper starter built on
  `\documentclass[conference]{IEEEtran}`.
- `.latexmkrc` adds the bundled IEEEtran class and bibliography directories to
  the local LaTeX search path.

Build notes:

- Build from this directory with `latexmk -pdf main.tex`.
- From the repo root, run `./scripts/sipaim_overleaf_sync.sh compile` to build
  `main.pdf` and refresh the tracked advisor-facing `sipaim2026.pdf`.
- On Overleaf, start from the official IEEE conference template or upload this
  folder and ensure `IEEEtran` is available.
- Remove all placeholder/template text before submission.
- Keep generated build artifacts out of git.
- Sync this folder to Overleaf only through
  `./scripts/sipaim_overleaf_sync.sh`; do not push the whole repo to Overleaf.
