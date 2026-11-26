"""Build college/27_LITERATURE_REVIEW.pdf from college/27_LITERATURE_REVIEW.md.

Usage (repo root):  venv/bin/python college/make_literature_review_pdf.py     (needs: pip install reportlab)
Supports the Markdown subset used in the review: #/##/### headings, paragraphs, - and 1. lists, pipe tables,
**bold**, *italic*, `code` and --- rules.
"""
import re
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdf_style import *  # noqa: F403  (styles, story and helpers)

SRC = Path(__file__).with_name("27_LITERATURE_REVIEW.md")
OUT = SRC.with_suffix(".pdf")
WIDTH = 16.4


def inline(text):
    t = escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r'<font name="Mono">\1</font>', t)
    return t


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table(lines):
    rows = [cells(l) for l in lines if not re.fullmatch(r"\|?[\s:|-]+\|?", l.strip())]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    # column widths proportional to (capped) content length
    plain = [[re.sub(r"[*`]", "", c) for c in r] for r in rows]
    # minimum width keeps the longest word (and short numeric columns) unbroken
    floor = [max(len(w) for r in plain for w in (r[i].split() or [""])) * 0.19 + 0.45 for i in range(n)]
    weight = [max(min(len(r[i]), 60) for r in plain) + 4 for i in range(n)]
    spare = max(WIDTH - sum(floor), 0)
    widths = [f + spare * w / sum(weight) for f, w in zip(floor, weight)]
    widths = [w * WIDTH / sum(widths) for w in widths]
    T([[inline(c) for c in r] for r in rows], widths)


def main():
    lines = SRC.read_text().splitlines()
    title = lines[0].lstrip("# ").strip()

    # cover
    story.append(Spacer(1, 5 * cm))
    P(inline(title.split(":")[0]), "title")
    story.append(Spacer(1, 0.4 * cm))
    P(inline(title.split(":", 1)[1].strip() if ":" in title else ""), "subtitle")
    story.append(Spacer(1, 0.4 * cm))
    P("A thematic review of 54 papers (2023–2027) · November 2026", "subtitle")
    story.append(PageBreak())
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-Bold", fontSize=10, leading=14),
                       ParagraphStyle("t1", fontName="Sans", fontSize=8.5, leading=11, leftIndent=16, textColor=MUTED)]
    P("Contents", "h1")
    story.append(toc)
    story.append(PageBreak())

    i, para = 1, []

    def flush():
        if para:
            P(inline(" ".join(para)))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s or s == "---":
            flush()
        elif s.startswith("### "):
            flush()
            P(f"<b>{inline(s[4:])}</b>", "body")
        elif s.startswith("## "):
            flush()
            H1(inline(s[3:]))
        elif s.startswith("# "):
            flush()
        elif s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table(block)
            continue
        elif re.match(r"^(- |\d+\. )", s):
            flush()
            while i < len(lines) and re.match(r"^\s*(- |\d+\. )", lines[i]):
                m = re.match(r"^\s*(- |(\d+)\. )(.*)", lines[i])
                bullet = "•" if m.group(1) == "- " else f"{m.group(2)}."
                story.append(Paragraph(inline(m.group(3)), S["bullet"], bulletText=bullet))
                i += 1
            story.append(Spacer(1, 4))
            continue
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            flush()
            P(inline(s.strip("*")), "small")
            story.append(Spacer(1, 4))
        else:
            para.append(s)
        i += 1
    flush()
    build(OUT, "Literature Review: Audio Deepfake Detection", "Thematic review of the 54 papers in papers/")


if __name__ == "__main__":
    main()
