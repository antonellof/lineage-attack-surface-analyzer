#!/usr/bin/env python3
"""Build the reading PDF and a portable LaTeX bundle from one manuscript.

Pandoc citeproc resolves references.bib using the local IEEE CSL style. The
resulting LaTeX contains its rendered bibliography and needs no BibTeX pass.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "PAPER.md"
OUT = HERE / "build" / "arxiv"
GITHUB = "https://github.com/antonellof/lineage-attack-surface-analyzer"


def split_front_matter(md: str) -> tuple[str, str, str, str, str]:
    lines = md.splitlines()
    title_idx = next(i for i, line in enumerate(lines) if line.startswith("# "))
    abstract_idx = next(i for i, line in enumerate(lines) if line.strip() == "## Abstract")
    author = next(line.strip() for line in lines[title_idx + 1:abstract_idx] if line.strip())
    next_idx = next(i for i in range(abstract_idx + 1, len(lines)) if lines[i].startswith("## "))
    abstract = "\n".join(lines[abstract_idx + 1:next_idx]).strip()
    body_idx = next(i for i, line in enumerate(lines) if re.match(r"## 1\. ", line))
    date_match = re.search(r"^Date:\s*(.+)$", md, re.MULTILINE)
    date = date_match.group(1).strip() if date_match else ""
    return lines[title_idx][2:].strip(), author, date, abstract, "\n".join(lines[body_idx:])


def yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_metadata_md(title: str, author: str, date: str, abstract: str, body: str,
                      email: str = "", repository: str = GITHUB) -> str:
    rows = [r"\normalsize " + author,
            r"\footnotesize Code and data: \nolinkurl{" + repository + "}"]
    if email:
        rows.append(r"\small \nolinkurl{" + email + "}")
    author_tex = r"\begin{tabular}{c}" + r"\\[3pt]".join(rows) + r"\end{tabular}"
    author_lines = ["  `" + author_tex + "`{=latex}"]
    return "\n".join([
        "---", f'title: "{yaml_escape(title)}"',
        "author: |", *author_lines,
        f'date: "{yaml_escape(date)}"', "abstract: |",
        *[f"  {line}" for line in abstract.splitlines()], "---", "", body, "",
    ])


# Latin Modern produces one ligature glyph in this build, U+FB00 for "ff". The
# fi, fl, ffi and ffl combinations are not ligatured here, so nothing else is
# mapped: an unverified mapping would corrupt the text it is meant to repair.
PDF_LIGATURES = {"\uFB00": "ff"}


def expand_pdf_ligatures(text: str) -> str:
    """Undo the ff ligature glyph that Latin Modern leaves in the PDF text layer.

    Rendering is correct either way; extraction is not. Copy, search and screen
    readers see the codepoint instead of the two letters, so the searchable text
    companion is normalized while the PDF itself is left untouched.
    """
    return "".join(PDF_LIGATURES.get(c, c) for c in text)


def ligature_count(text: str) -> int:
    return sum(c in PDF_LIGATURES for c in text)


def write_text_companion(pdf: Path, out: Path) -> int:
    """Write a searchable text copy of the reading PDF; return the ligature count."""
    if not shutil.which("pdftotext"):
        return 0
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    out.write_text(expand_pdf_ligatures(text))
    return ligature_count(text)


def run_pandoc(meta_md: Path, tex_out: Path, pdf_out: Path | None) -> None:
    common = [
        "pandoc", str(meta_md), "--standalone", "--citeproc", "--fail-if-warnings", "--pdf-engine=tectonic",
        "--bibliography", str(HERE / "references.bib"),
        "--csl", str(HERE / "ieee.csl"),
        "--include-in-header", str(HERE / "formatting.tex"),
        "--resource-path", str(HERE),
        "-V", "documentclass=article", "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt", "-V", "colorlinks=true",
        "-V", "linkcolor=black", "-V", "urlcolor=blue",
    ]
    subprocess.run(common + ["-o", str(tex_out)], check=True)
    if pdf_out:
        subprocess.run(common + ["-o", str(pdf_out)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--reading-copy", type=Path, help="copy the compiled PDF to this path")
    args = parser.parse_args()
    if args.reading_copy and not args.pdf:
        parser.error("--reading-copy requires --pdf")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    # Bundle only the figures actually used by the current manuscript.
    md = SRC.read_text()
    for filename in re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", md):
        shutil.copy(HERE / filename, OUT / filename)
    for filename in ("references.bib", "ieee.csl", "formatting.tex"):
        shutil.copy(HERE / filename, OUT / filename)
    metadata = OUT / "_paper.md"
    email = re.search(r"^Email:\s*(\S+)$", md, re.MULTILINE)
    repository = re.search(r"^Code and data:\s*(https://\S+)$", md, re.MULTILINE)
    metadata.write_text(build_metadata_md(*split_front_matter(md),
                                         email=email.group(1) if email else "",
                                         repository=repository.group(1) if repository else GITHUB))
    pdf = OUT / "paper.pdf" if args.pdf else None
    run_pandoc(metadata, OUT / "paper.tex", pdf)
    metadata.unlink()
    if pdf is not None:
        replaced = write_text_companion(pdf, OUT / "paper.txt")
        if replaced:
            print(f"text companion: expanded {replaced} ligature glyphs into paper.txt")
    if args.reading_copy:
        shutil.copy(pdf, args.reading_copy)
    (OUT / "README.txt").write_text(
        "Generated from paper/PAPER.md. Main source: paper.tex.\n"
        "Citations are rendered inline by Pandoc citeproc; no BibTeX pass is needed.\n"
        "references.bib and ieee.csl are included for provenance and rebuilding.\n"
        "Compile paper.tex with XeLaTeX/Tectonic in this directory; pdflatex is not validated.\n"
        "paper.txt is a searchable text copy of the PDF. Latin Modern puts an ff ligature\n"
        " glyph in the PDF text layer, so the companion expands it for copy, search, and\n"
        " screen readers; the figures themselves are not in the text copy.\n"
        "Check the current submission rules of the selected venue before upload.\n"
        "Complete the independent evaluation and human audit before publication.\n"
    )
    print(f"wrote {OUT / 'paper.tex'}")
    if args.reading_copy:
        print(f"wrote {args.reading_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
