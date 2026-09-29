"""Build a reflowable personal-reading EPUB from the compiled INCISCOS draft.

Run from the repository root with ``.venv/bin/python paper/inciscos2026/make_epub.py``.
The crop coordinates below match the current ten-page review PDF. Update them
if a later revision moves the TikZ figures to different pages.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent.parent
PANDOC = REPO / ".venv/pandoc-local/usr/bin/pandoc"
PANDOC_DATA = REPO / ".venv/pandoc-local/usr/share/pandoc/data"
SOURCE = PAPER / "main.tex"
PDF = PAPER / "inciscos2026.pdf"
OUTPUT = PAPER / "inciscos2026_kindle.epub"

# PDF page, then x0, y0, x1, y1 in points. Captions are set as reflowable text.
FIGURE_CROPS = {
    "fig:dataset-examples": (3, 35, 38, 560, 390),
    "fig:data-split": (4, 35, 40, 560, 282),
    "fig:vae-pipeline": (5, 35, 40, 560, 133),
    "fig:residual-blocks": (5, 35, 168, 560, 362),
    "fig:vae-architectures": (6, 35, 40, 560, 245),
}

FIGURES = {
    "fig:dataset-examples": (1, None),
    "fig:data-split": (2, None),
    "fig:vae-pipeline": (3, None),
    "fig:residual-blocks": (4, None),
    "fig:vae-architectures": (5, None),
    "fig:reconstructions": (6, PAPER / "figures/reconstructions_fixed9.png"),
    "fig:reconstruction-boxplots": (7, PAPER / "figures/reconstruction_boxplots.png"),
    "fig:downstream": (8, PAPER / "figures/downstream_results.png"),
}

TABLES = {
    "tab:task-populations": 1,
    "tab:architectures": 2,
    "tab:reconstruction": 3,
    "tab:tissue": 4,
    "tab:wsi-overall": 5,
    "tab:wsi-class": 6,
}


def braced(text: str, start: int) -> tuple[str, int]:
    """Return the contents and ending offset of a balanced braced group."""
    assert text[start] == "{"
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ValueError("Unclosed LaTeX group")


def caption_and_body(block: str) -> tuple[str, str]:
    match = re.search(r"\\caption\s*\{", block)
    if not match:
        raise ValueError("Missing figure or table caption")
    caption, end = braced(block, match.end() - 1)
    return caption, block[: match.start()] + block[end:]


def crop_pdf_figure(page: int, box: tuple[int, int, int, int], output: Path) -> None:
    with tempfile.TemporaryDirectory() as render_dir:
        prefix = Path(render_dir) / "page"
        subprocess.run(
            ["pdftoppm", "-f", str(page), "-l", str(page), "-r", "220",
             "-png", "-singlefile", str(PDF), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        with Image.open(prefix.with_suffix(".png")) as image:
            sx, sy = image.width / 595.276, image.height / 841.89
            x0, y0, x1, y1 = box
            image.crop((round(x0 * sx), round(y0 * sy), round(x1 * sx),
                        round(y1 * sy))).save(output)


def build() -> None:
    if not PANDOC.is_file() or not PANDOC_DATA.is_dir():
        raise SystemExit("Local Pandoc is missing from .venv/pandoc-local")
    if not PDF.is_file():
        raise SystemExit("Compile inciscos2026.pdf before building the EPUB")

    source = SOURCE.read_text()
    with tempfile.TemporaryDirectory(prefix="inciscos-epub-") as work_str:
        work = Path(work_str)
        for label, (page, *box) in FIGURE_CROPS.items():
            crop_pdf_figure(page, tuple(box), work / f"{label[4:]}.png")

        def figure_replacement(match: re.Match[str]) -> str:
            block = match.group(0)
            label_match = re.search(r"\\label\{(fig:[^}]+)\}", block)
            if not label_match:
                raise ValueError("Unlabeled figure")
            label = label_match.group(1)
            number, existing = FIGURES[label]
            caption, _ = caption_and_body(block)
            image = existing or work / f"{label[4:]}.png"
            return ("\n\\begin{center}\n"
                    f"\\includegraphics[width=0.95\\textwidth]{{{image}}}\n\n"
                    f"\\textbf{{Figure {number}.}} {caption}\n"
                    "\\end{center}\n")

        source = re.sub(
            r"\\begin\{figure\*?\}(?:\[[^]]*\])?.*?\\end\{figure\*?\}",
            figure_replacement, source, flags=re.DOTALL,
        )

        def table_replacement(match: re.Match[str]) -> str:
            block = match.group(0)
            label_match = re.search(r"\\label\{(tab:[^}]+)\}", block)
            if not label_match:
                raise ValueError("Unlabeled table")
            label = label_match.group(1)
            caption, body = caption_and_body(block)
            body = re.sub(r"\\label\{tab:[^}]+\}", "", body, count=1)
            return f"\n\\textbf{{Table {TABLES[label]}.}} {caption}\n\n{body}\n"

        source = re.sub(
            r"\\begin\{table\*?\}(?:\[[^]]*\])?.*?\\end\{table\*?\}",
            table_replacement, source, flags=re.DOTALL,
        )

        def autoref(match: re.Match[str]) -> str:
            label = match.group(1)
            if label in FIGURES:
                return f"Figure {FIGURES[label][0]}"
            if label in TABLES:
                return f"Table {TABLES[label]}"
            if label == "eq:equivariance":
                return "Equation 1"
            raise ValueError(f"Unknown cross-reference: {label}")

        source = re.sub(r"\\autoref\{([^}]+)\}", autoref, source)
        source = re.sub(r"\\label\{eq:[^}]+\}", "", source)

        prepared = work / "kindle.tex"
        prepared.write_text(source)
        subprocess.run(
            [str(PANDOC), f"--data-dir={PANDOC_DATA}", str(prepared),
             "--from=latex", "--to=epub3", f"--resource-path={PAPER}:{work}",
             f"--bibliography={PAPER / 'references.bib'}", "--citeproc", "--mathml",
             "--metadata=lang:en", f"--output={OUTPUT}"],
            check=True,
            cwd=PAPER,
        )
    print(OUTPUT)


if __name__ == "__main__":
    build()
