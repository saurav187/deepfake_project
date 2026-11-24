"""Shared ReportLab styles and helpers for the college PDFs."""
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    PageBreak,  # noqa: F401  (re-exported to the PDF scripts)
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: F401  (re-exported)

FONTS = Path("/usr/share/fonts/TTF")
pdfmetrics.registerFont(TTFont("Sans", FONTS / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONTS / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Italic", FONTS / "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("Sans-BoldItalic", FONTS / "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("Mono", FONTS / "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-BoldItalic")

INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#52606d")
ACCENT = colors.HexColor("#1d4ed8")
SOFT = colors.HexColor("#eef2ff")
LINE = colors.HexColor("#cbd2d9")
GREEN, RED, AMBER = colors.HexColor("#15803d"), colors.HexColor("#b91c1c"), colors.HexColor("#b45309")
NOTE_BG, WARN_BG = colors.HexColor("#ecfdf5"), colors.HexColor("#fff7ed")

base = dict(fontName="Sans", textColor=INK, leading=14.5, fontSize=10)
S = {
    "body": ParagraphStyle("body", spaceAfter=6, **base),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=17, leading=21, textColor=ACCENT, spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=10, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.5, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.8, leading=11.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.8, leading=11.2, textColor=INK),
    "bullet": ParagraphStyle("bullet", leftIndent=14, bulletIndent=3, spaceAfter=3, **base),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=26, leading=32, textColor=INK, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=18, textColor=MUTED, alignment=TA_CENTER),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.8, leading=10, textColor=INK),
}
story = []


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    story.append(CondPageBreak(9 * cm))
    p = Paragraph(text, S["h1"])
    p.toc_level = 0
    story.append(p)


def H2(text):
    story.append(CondPageBreak(4 * cm))
    p = Paragraph(text, S["h2"])
    p.toc_level = 1
    story.append(p)


def B(*items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))
    story.append(Spacer(1, 4))


def box(text, kind="note"):
    bg, edge, label = {"note": (NOTE_BG, GREEN, "In simple words"), "warn": (WARN_BG, AMBER, "Why it matters"),
                       "file": (SOFT, ACCENT, "Where in the code"),
                       "draft": (SOFT, ACCENT, "Draft")}[kind]
    t = Table([[Paragraph(f"<b>{label}:</b> {text}", S["cell"])]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([t, Spacer(1, 8)])


def T(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb" if header and i == 0 else "cell"]) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), SOFT))
    t.setStyle(TableStyle(style))
    story.extend([t, Spacer(1, 9)])


def CODE(text):
    t = Table([[Preformatted(text.strip("\n"), S["code"])]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f7fa")),
                           ("BOX", (0, 0), (-1, -1), 0.4, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.extend([t, Spacer(1, 8)])


def arrow_down(d, x, y1, y2):
    d.add(Line(x, y1, x, y2 + 5, strokeColor=MUTED, strokeWidth=1.2))
    d.add(Polygon([x - 4, y2 + 6, x + 4, y2 + 6, x, y2], fillColor=MUTED, strokeColor=MUTED))


def flow(steps, width=16.4 * cm, box_h=30, gap=14, title=None):
    """Vertical flow chart: list of (label, detail, colour)."""
    h = len(steps) * box_h + (len(steps) - 1) * gap + (18 if title else 4)
    d = Drawing(width, h)
    y = h - (18 if title else 4)
    if title:
        d.add(String(0, h - 12, title, fontName="Sans-Bold", fontSize=9.5, fillColor=MUTED))
    for i, (label, detail, col) in enumerate(steps):
        y0 = y - box_h
        d.add(Rect(0, y0, width, box_h, rx=5, ry=5, fillColor=col, strokeColor=LINE, strokeWidth=0.6))
        d.add(String(10, y0 + box_h - 13, label, fontName="Sans-Bold", fontSize=9.2, fillColor=INK))
        d.add(String(10, y0 + 6, detail, fontName="Sans", fontSize=8, fillColor=MUTED))
        if i < len(steps) - 1:
            arrow_down(d, width / 2, y0, y0 - gap)
        y = y0 - gap
    story.extend([d, Spacer(1, 10)])



class Doc(BaseDocTemplate):
    footer = ""

    def __init__(self, path, title, subject=""):
        super().__init__(str(path), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.0 * cm,
                         bottomMargin=2.0 * cm, title=title, author="deepfake_project", subject=subject)
        Doc.footer = title
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self.decorate)])

    @staticmethod
    def decorate(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.3 * cm, 1.2 * cm, Doc.footer)
        canvas.drawRightString(A4[0] - 2.3 * cm, 1.2 * cm, f"{doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(2.3 * cm, 1.5 * cm, A4[0] - 2.3 * cm, 1.5 * cm)
        canvas.restoreState()

    def afterFlowable(self, f):
        level = getattr(f, "toc_level", None)
        if level is not None:
            self.notify("TOCEntry", (level, f.getPlainText(), self.page))




def build(path, title, subject=""):
    Doc(path, title, subject).multiBuild(story)
    print(f"Wrote {path}")
