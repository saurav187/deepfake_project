"""ReportLab layout, typography, Roman->Arabic page numbering, TOC, and vector diagrams
matching report/Template.pdf (Cluster Innovation Centre, University of Delhi)."""
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

NOTO = Path("/usr/share/fonts/noto")
DEJAVU = Path("/usr/share/fonts/TTF")

pdfmetrics.registerFont(TTFont("Serif", NOTO / "NotoSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Bold", NOTO / "NotoSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Italic", NOTO / "NotoSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Serif-BoldItalic", NOTO / "NotoSerif-BoldItalic.ttf"))
pdfmetrics.registerFontFamily(
    "Serif",
    normal="Serif",
    bold="Serif-Bold",
    italic="Serif-Italic",
    boldItalic="Serif-BoldItalic",
)
pdfmetrics.registerFont(TTFont("Mono", DEJAVU / "DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Sans", DEJAVU / "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", DEJAVU / "DejaVuSans-Bold.ttf"))

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")
BORDER = colors.HexColor("#9ca3af")
GRID = colors.HexColor("#cbd5e1")
HEAD_BG = colors.HexColor("#edf2f7")
ALT_BG = colors.HexColor("#f8fafc")
ACCENT = colors.HexColor("#1e3a8a")
SOFT_BLUE = colors.HexColor("#eff6ff")
SOFT_AMBER = colors.HexColor("#fef3c7")
SOFT_GREEN = colors.HexColor("#ecfdf5")

PAGE_W, PAGE_H = A4
MARGIN_L = 2.54 * cm
MARGIN_R = 2.54 * cm
MARGIN_T = 2.4 * cm
MARGIN_B = 2.4 * cm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R  # ~15.92 cm

STYLES = {
    "cover_title": ParagraphStyle(
        "cover_title", fontName="Serif", fontSize=20, leading=26,
        alignment=TA_CENTER, textColor=INK, spaceAfter=10,
    ),
    "cover_inst": ParagraphStyle(
        "cover_inst", fontName="Serif-Italic", fontSize=14.5, leading=20,
        alignment=TA_CENTER, textColor=INK, spaceAfter=8,
    ),
    "cover_body": ParagraphStyle(
        "cover_body", fontName="Serif", fontSize=12, leading=17,
        alignment=TA_CENTER, textColor=INK, spaceAfter=4,
    ),
    "cover_bold": ParagraphStyle(
        "cover_bold", fontName="Serif-Bold", fontSize=12, leading=17,
        alignment=TA_CENTER, textColor=INK, spaceAfter=4,
    ),
    "front_h1": ParagraphStyle(
        "front_h1", fontName="Serif", fontSize=18, leading=24,
        alignment=TA_CENTER, textColor=INK, spaceBefore=6, spaceAfter=18,
    ),
    "chapter": ParagraphStyle(
        "chapter", fontName="Serif-Bold", fontSize=15, leading=20,
        alignment=TA_LEFT, textColor=INK, spaceBefore=4, spaceAfter=12,
    ),
    "h2": ParagraphStyle(
        "h2", fontName="Serif-Bold", fontSize=12.2, leading=16.5,
        alignment=TA_LEFT, textColor=INK, spaceBefore=12, spaceAfter=5,
    ),
    "h3": ParagraphStyle(
        "h3", fontName="Serif-Bold", fontSize=11, leading=15,
        alignment=TA_LEFT, textColor=INK, spaceBefore=9, spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "body", fontName="Serif", fontSize=10.6, leading=15.6,
        alignment=TA_JUSTIFY, textColor=INK, spaceAfter=7,
    ),
    "body_left": ParagraphStyle(
        "body_left", fontName="Serif", fontSize=10.6, leading=15.6,
        alignment=TA_LEFT, textColor=INK, spaceAfter=7,
    ),
    "bullet": ParagraphStyle(
        "bullet", fontName="Serif", fontSize=10.4, leading=15.0,
        alignment=TA_JUSTIFY, textColor=INK, leftIndent=16, bulletIndent=5, spaceAfter=4,
    ),
    "cell": ParagraphStyle(
        "cell", fontName="Serif", fontSize=8.8, leading=12.0,
        alignment=TA_LEFT, textColor=INK,
    ),
    "cellb": ParagraphStyle(
        "cellb", fontName="Serif-Bold", fontSize=8.8, leading=12.0,
        alignment=TA_LEFT, textColor=INK,
    ),
    "caption": ParagraphStyle(
        "caption", fontName="Serif-Italic", fontSize=9.2, leading=12.5,
        alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=10,
    ),
    "ref": ParagraphStyle(
        "ref", fontName="Serif", fontSize=9.2, leading=13.0,
        alignment=TA_JUSTIFY, textColor=INK, leftIndent=22, firstLineIndent=-22, spaceAfter=4,
    ),
    "formula": ParagraphStyle(
        "formula", fontName="Mono", fontSize=9.0, leading=13.0,
        alignment=TA_CENTER, textColor=INK, spaceBefore=4, spaceAfter=8,
    ),
    "sig_left": ParagraphStyle(
        "sig_left", fontName="Serif-Bold", fontSize=10.5, leading=14.5, alignment=TA_LEFT, textColor=INK,
    ),
    "sig_right": ParagraphStyle(
        "sig_right", fontName="Serif-Bold", fontSize=10.5, leading=14.5, alignment=TA_RIGHT, textColor=INK,
    ),
}


def to_roman(n: int) -> str:
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


class ResetPageNumber(Flowable):
    """Switch numbering mode ('roman' or 'arabic') and reset logical page counter to 1."""
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.width = 0
        self.height = 0

    def draw(self):
        pass


class HorizontalRule(Flowable):
    def __init__(self, width=CONTENT_W, thickness=0.8, color=INK):
        super().__init__()
        self.width = width
        self.height = thickness + 2
        self.thickness = thickness
        self.color = color

    def draw(self):
        self.canv.saveState()
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 1, self.width, 1)
        self.canv.restoreState()


class AcademicDoc(BaseDocTemplate):
    def __init__(self, path: Path, title: str):
        super().__init__(
            str(path), pagesize=A4,
            leftMargin=MARGIN_L, rightMargin=MARGIN_R,
            topMargin=MARGIN_T, bottomMargin=MARGIN_B,
            title=title, author="Saurav Kumar, Abhijeet Kumar, Rupesh Bhagat",
            subject="Semester VII Research Project Report — Cluster Innovation Centre, University of Delhi",
        )
        self._roman_page = 0
        self._arabic_page = 0
        frame = Frame(MARGIN_L, MARGIN_B, CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B, id="main")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._on_cover),
            PageTemplate(id="roman", frames=[frame], onPage=self._on_roman),
            PageTemplate(id="arabic", frames=[frame], onPage=self._on_arabic),
        ])

    def handle_documentBegin(self):
        super().handle_documentBegin()
        self._roman_page = 0
        self._arabic_page = 0

    def _on_cover(self, canvas, doc):
        pass

    def _on_roman(self, canvas, doc):
        doc._roman_page += 1
        canvas.saveState()
        canvas.setFont("Serif", 10)
        canvas.setFillColor(INK)
        canvas.drawCentredString(PAGE_W / 2, 1.35 * cm, to_roman(doc._roman_page))
        canvas.restoreState()

    def _on_arabic(self, canvas, doc):
        doc._arabic_page += 1
        canvas.saveState()
        canvas.setFont("Serif", 10)
        canvas.setFillColor(INK)
        canvas.drawCentredString(PAGE_W / 2, 1.35 * cm, str(doc._arabic_page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        level = getattr(flowable, "toc_level", None)
        if level is not None and self.pageTemplate.id == "arabic":
            text = flowable.getPlainText()
            self.notify("TOCEntry", (level, text, self._arabic_page))


def make_toc():
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [
        ParagraphStyle("toc0", fontName="Serif-Bold", fontSize=10.2, leading=15, leftIndent=0, firstLineIndent=0, spaceBefore=3),
        ParagraphStyle("toc1", fontName="Serif", fontSize=9.8, leading=14, leftIndent=16, firstLineIndent=0),
        ParagraphStyle("toc2", fontName="Serif", fontSize=9.4, leading=13, leftIndent=34, firstLineIndent=0),
    ]
    return toc


def add_chapter(story, text, page_break=True):
    if page_break:
        story.append(PageBreak())
    p = Paragraph(text, STYLES["chapter"])
    p.toc_level = 0
    story.append(p)


def add_h2(story, text):
    story.append(CondPageBreak(4.2 * cm))
    p = Paragraph(text, STYLES["h2"])
    p.toc_level = 1
    story.append(p)


def add_h3(story, text):
    story.append(CondPageBreak(3.2 * cm))
    p = Paragraph(text, STYLES["h3"])
    p.toc_level = 2
    story.append(p)


def add_p(story, text, style="body"):
    story.append(Paragraph(text, STYLES[style]))


def add_bullets(story, items):
    for item in items:
        story.append(Paragraph(item, STYLES["bullet"], bulletText="•"))
    story.append(Spacer(1, 4))


def add_table(story, rows, widths_cm, caption=None):
    data = [
        [Paragraph(str(cell), STYLES["cellb" if r == 0 else "cell"]) for cell in row]
        for r, row in enumerate(rows)
    ]
    t = Table(data, colWidths=[w * cm for w in widths_cm], repeatRows=1)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), ALT_BG))
    t.setStyle(TableStyle(style))
    story.append(Spacer(1, 3))
    story.append(t)
    if caption:
        story.append(Paragraph(caption, STYLES["caption"]))
    else:
        story.append(Spacer(1, 8))


def add_figure(story, img_path: Path, caption: str, width_cm=15.4):
    from PIL import Image as PILImage
    with PILImage.open(img_path) as im:
        w, h = im.size
    aspect = h / w
    height_cm = width_cm * aspect
    story.append(CondPageBreak((height_cm + 2.2) * cm))
    story.append(Spacer(1, 4))
    img = Image(str(img_path), width=width_cm * cm, height=height_cm * cm)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Paragraph(caption, STYLES["caption"]))


# ------------------------------------------------------------------ Vector Diagrams (Ch. 5)
def _arrow(d, x1, y1, x2, y2, col=MUTED):
    d.add(Line(x1, y1, x2, y2, strokeColor=col, strokeWidth=1.1))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    sz = 5.5
    p1 = (x2 - sz * math.cos(ang - 0.45), y2 - sz * math.sin(ang - 0.45))
    p2 = (x2 - sz * math.cos(ang + 0.45), y2 - sz * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=col, strokeColor=col))


def _box(d, x, y, w, h, title, sub="", fill=SOFT_BLUE, edge=ACCENT):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=edge, strokeWidth=0.85))
    if sub:
        d.add(String(x + w / 2, y + h - 14, title, fontName="Sans-Bold", fontSize=8.2, fillColor=INK, textAnchor="middle"))
        d.add(String(x + w / 2, y + 6, sub, fontName="Sans", fontSize=7.0, fillColor=MUTED, textAnchor="middle"))
    else:
        d.add(String(x + w / 2, y + h / 2 - 3, title, fontName="Sans-Bold", fontSize=8.2, fillColor=INK, textAnchor="middle"))


def draw_architecture_diagram():
    w, h = CONTENT_W, 9.6 * cm
    d = Drawing(w, h)
    cx = w / 2
    # Row 1: Client
    _box(d, cx - 4.8 * cm, 8.0 * cm, 9.6 * cm, 1.3 * cm,
         "Browser Client (React 18 + Vite + TypeScript)",
         "DropZone, Web Audio waveform, local playback, polling UI", fill=SOFT_BLUE)
    # Row 2: FastAPI
    _box(d, cx - 5.9 * cm, 5.75 * cm, 11.8 * cm, 1.35 * cm,
         "FastAPI Backend (app/backend/main.py)",
         "Streaming multipart upload (<=50 MB), ffprobe (<=10 min), CSP & no-store", fill=HEAD_BG, edge=BORDER)
    # Row 3: Worker pool
    _box(d, cx - 5.5 * cm, 3.6 * cm, 11.0 * cm, 1.3 * cm,
         "In-Memory Job Queue & ThreadPoolExecutor (jobs.py)",
         "60 min TTL expiry, cancellation, zero audio storage or logging", fill=SOFT_AMBER, edge=colors.HexColor("#b45309"))
    # Row 4: Detector interface + 4 backends
    bw = 3.65 * cm
    gap = 0.44 * cm
    backends = [
        ("MockDetector", "DETECTOR_BACKEND=mock", ALT_BG, BORDER),
        ("LegacyDetector", "Gen 1 CNN-LSTM", ALT_BG, BORDER),
        ("Release v1", "Frozen XLS-R 300M", SOFT_GREEN, colors.HexColor("#15803d")),
        ("V2-90 / V3", "C2+C3 Fine-Tuned", SOFT_BLUE, ACCENT),
    ]
    for i, (t, s, fl, ed) in enumerate(backends):
        bx = i * (bw + gap)
        _box(d, bx, 0.35 * cm, bw, 1.35 * cm, t, s, fill=fl, edge=ed)
        _arrow(d, cx, 2.75 * cm, bx + bw / 2, 1.73 * cm)

    _arrow(d, cx, 8.0 * cm, cx, 7.12 * cm)
    _arrow(d, cx, 5.75 * cm, cx, 4.92 * cm)
    d.add(Line(cx, 3.6 * cm, cx, 2.75 * cm, strokeColor=MUTED, strokeWidth=1.1))
    d.add(String(cx + 0.2 * cm, 7.46 * cm, "POST /api/analyze & GET /api/jobs/{id}", fontName="Sans", fontSize=6.8, fillColor=MUTED))
    # Clean pill box for detector contract label
    d.add(Rect(cx - 4.35 * cm, 2.55 * cm, 8.7 * cm, 0.52 * cm, rx=3, ry=3, fillColor=colors.white, strokeColor=ACCENT, strokeWidth=0.6))
    d.add(String(cx, 2.71 * cm, "Raw bytes -> Detector.analyze(data, filename)", fontName="Sans-Bold", fontSize=7.2, fillColor=ACCENT, textAnchor="middle"))
    return d


def draw_dfd_diagram():
    w, h = CONTENT_W, 5.4 * cm
    d = Drawing(w, h)
    # Top flow (left to right)
    boxes = [
        (0.1 * cm, 3.1 * cm, 2.9 * cm, 1.5 * cm, "1. Audio File", "WAV/MP3/M4A/MP4", SOFT_BLUE, ACCENT),
        (3.6 * cm, 3.1 * cm, 3.2 * cm, 1.5 * cm, "2. Stream Parser", "In-RAM size cap", HEAD_BG, BORDER),
        (7.4 * cm, 3.1 * cm, 3.2 * cm, 1.5 * cm, "3. ffprobe Gate", "Duration <= 600 s", SOFT_AMBER, colors.HexColor("#b45309")),
        (11.2 * cm, 3.1 * cm, 4.5 * cm, 1.5 * cm, "4. Canonical Decode", "16 kHz mono, trim, RMS", SOFT_GREEN, colors.HexColor("#15803d")),
    ]
    for x, y, bw, bh, t, s, fl, ed in boxes:
        _box(d, x, y, bw, bh, t, s, fill=fl, edge=ed)
    _arrow(d, 3.0 * cm, 3.85 * cm, 3.58 * cm, 3.85 * cm)
    _arrow(d, 6.8 * cm, 3.85 * cm, 7.38 * cm, 3.85 * cm)
    _arrow(d, 10.6 * cm, 3.85 * cm, 11.18 * cm, 3.85 * cm)

    # Bottom return flow (right to left)
    _box(d, 11.2 * cm, 0.5 * cm, 4.5 * cm, 1.5 * cm, "5. SSL Scoring", "C2+C3 z-mean / v1 heads", SOFT_GREEN, colors.HexColor("#15803d"))
    _box(d, 5.6 * cm, 0.5 * cm, 4.8 * cm, 1.5 * cm, "6. Calibrated Decision", "Verdict + segments JSON", HEAD_BG, BORDER)
    _box(d, 0.1 * cm, 0.5 * cm, 4.8 * cm, 1.5 * cm, "7. Browser Display", "Card + timeline + local play", SOFT_BLUE, ACCENT)

    _arrow(d, 13.45 * cm, 3.1 * cm, 13.45 * cm, 2.02 * cm)
    _arrow(d, 11.2 * cm, 1.25 * cm, 10.42 * cm, 1.25 * cm)
    _arrow(d, 5.6 * cm, 1.25 * cm, 4.92 * cm, 1.25 * cm)
    d.add(String(13.6 * cm, 2.45 * cm, "Temp file deleted", fontName="Sans", fontSize=6.8, fillColor=MUTED))
    return d


def draw_sequence_diagram():
    w, h = CONTENT_W, 8.2 * cm
    d = Drawing(w, h)
    actors = [
        (1.9 * cm, "Browser UI"),
        (5.9 * cm, "FastAPI Server"),
        (10.0 * cm, "Job Worker Pool"),
        (14.0 * cm, "Detector Model"),
    ]
    for x, name in actors:
        _box(d, x - 1.65 * cm, 7.1 * cm, 3.3 * cm, 0.85 * cm, name, fill=HEAD_BG, edge=BORDER)
        d.add(Line(x, 7.1 * cm, x, 0.4 * cm, strokeColor=GRID, strokeWidth=1.0, strokeDashArray=[3, 3]))

    steps = [
        (6.4 * cm, 1.9, 5.9, "1. POST /api/analyze (multipart bytes)"),
        (5.6 * cm, 5.9, 5.9, "2. Validate ext, size <=50 MB, ffprobe <=600 s"),
        (4.8 * cm, 5.9, 10.0, "3. Enqueue job_id (clear bytes once popped)"),
        (4.1 * cm, 5.9, 1.9, "4. HTTP 202 {job_id, status: queued}"),
        (3.3 * cm, 10.0, 14.0, "5. detector.analyze(data, filename)"),
        (2.5 * cm, 14.0, 10.0, "6. Return AnalysisResult (unlink temp file)"),
        (1.7 * cm, 1.9, 5.9, "7. Poll GET /api/jobs/{job_id} (1 s)"),
        (0.95 * cm, 5.9, 1.9, "8. JSON verdict + segment timeline (no audio)"),
    ]
    for y, x1, x2, label in steps:
        if x1 == x2:
            d.add(Rect(x1 * cm - 0.12 * cm, y - 0.15 * cm, 0.24 * cm, 0.35 * cm, fillColor=SOFT_AMBER, strokeColor=BORDER))
            d.add(String(x1 * cm + 0.25 * cm, y - 0.04 * cm, label, fontName="Sans", fontSize=7.2, fillColor=INK))
        else:
            _arrow(d, x1 * cm, y, x2 * cm, y, col=ACCENT if x1 < x2 else colors.HexColor("#15803d"))
            mx = (x1 + x2) / 2 * cm
            d.add(String(mx, y + 0.12 * cm, label, fontName="Sans", fontSize=7.1, fillColor=INK, textAnchor="middle"))
    return d
