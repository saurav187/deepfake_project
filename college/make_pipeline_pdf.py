"""Build college/4_HOW_THE_PIPELINE_WORKS.pdf, a plain-language walkthrough of the detector code.

Usage (repo root):  venv/bin/python college/make_pipeline_pdf.py     (needs: pip install reportlab)
"""
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
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = Path(__file__).with_name("4_HOW_THE_PIPELINE_WORKS.pdf")
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
                       "file": (SOFT, ACCENT, "Where in the code")}[kind]
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


def model_diagram():
    blue, green, grey = colors.HexColor("#e0f2fe"), colors.HexColor("#dcfce7"), colors.HexColor("#f1f5f9")
    flow([
        ("4 s window of prepared audio", "64,000 numbers (16,000 per second)", grey),
        ("XLS-R CNN front-end  [frozen]", "turns the waveform into 199 frames, one every 20 ms", blue),
        ("XLS-R: 24 transformer layers  [frozen, 315.4M parameters]", "each frame becomes 1,024 numbers; all 25 layer outputs are kept", blue),
        ("Layer mix  [trained]", "25 learned weights blend the 25 layer outputs into one (most weight on layers 4, 6, 10)", green),
        ("LayerNorm + projection  [trained]", "1,024 → 128 numbers per frame", green),
        ("Attentive statistics pooling  [trained]", "picks the frames that matter; 199 frames → 256 numbers (weighted mean + spread)", green),
        ("Small MLP classifier  [trained]", "256 → 128 → 1 number = the logit (head total: 182,939 parameters, ×3 seeds)", green),
    ], box_h=28, gap=8, title="Inside the model: one 4-second window → one logit")
    P("Each of the 3 heads gives one logit per window. The app and the test average the 3 heads, then all windows of the recording, "
      "then apply calibration (section 11) to get a probability and a verdict.", "small")


def prep_diagram():
    """Sketch of what prepare() does to a recording's level over time."""
    import math
    w, h = 16.4 * cm, 4.2 * cm
    d = Drawing(w, h)
    top, mid = 3.6 * cm, 2.2 * cm

    def wave(x0, x1, amp_fn, col):
        n = 220
        pts = []
        for i in range(n + 1):
            t = i / n
            x = x0 + (x1 - x0) * t
            pts.append((x, mid + amp_fn(t) * math.sin(i * 1.7)))
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            d.add(Line(xa, ya, xb, yb, strokeColor=col, strokeWidth=0.6))

    W = 7.4 * cm
    wave(0, W, lambda t: 0.06 * cm if t < 0.12 or t > 0.86 else 1.0 * cm * (0.6 + 0.4 * math.sin(t * 9) ** 2), MUTED)
    d.add(String(0, top + 0.2 * cm, "Before: silence + speech + silence", fontName="Sans-Bold", fontSize=8, fillColor=INK))
    for x, lab in ((0.03 * W, "silence"), (0.88 * W, "silence")):
        d.add(String(x, 0.55 * cm, lab, fontName="Sans", fontSize=6.8, fillColor=RED))
    d.add(Line(W + 0.4 * cm, mid, W + 1.4 * cm, mid, strokeColor=ACCENT, strokeWidth=1.4))
    d.add(Polygon([W + 1.4 * cm, mid + 4, W + 1.4 * cm, mid - 4, W + 1.6 * cm, mid], fillColor=ACCENT, strokeColor=ACCENT))
    X0 = W + 2.0 * cm
    W2 = 6.4 * cm

    def env(t):
        fade = 0.06
        ramp = 1.0 if fade < t < 1 - fade else (0.5 - 0.5 * math.cos(math.pi * min(t, 1 - t) / fade))
        return 1.15 * cm * ramp * (0.6 + 0.4 * math.sin(t * 9 + 1) ** 2)
    wave(X0, X0 + W2, env, ACCENT)
    d.add(String(X0, top + 0.2 * cm, "After: trimmed, edges cut, faded, level = −25 dBFS", fontName="Sans-Bold", fontSize=8, fillColor=INK))
    d.add(String(X0, 0.55 * cm, "fade-in 0.15 s", fontName="Sans", fontSize=6.8, fillColor=GREEN))
    d.add(String(X0 + W2 - 1.6 * cm, 0.55 * cm, "fade-out 0.15 s", fontName="Sans", fontSize=6.8, fillColor=GREEN))
    story.extend([d, Spacer(1, 8)])


# ---------------------------------------------------------------- content
story.append(Spacer(1, 4.5 * cm))
P("How the Deepfake Detector Works", "title")
story.append(Spacer(1, 0.5 * cm))
P("A plain-language walkthrough of the code, from raw audio files to the final, sealed test", "subtitle")
story.append(Spacer(1, 0.35 * cm))
P("Release v1 · branch detector-v2 · November 2026", "subtitle")
story.append(Spacer(1, 2.2 * cm))
T([["What you will learn", ""],
   ["1–2", "The big picture and the words used everywhere"],
   ["3–5", "Data: where it comes from, how it is checked, the shortcuts we found"],
   ["6–7", "Preprocessing and augmentation: what happens to every recording"],
   ["8–10", "The model and how it is trained and chosen"],
   ["11–13", "Calibration, freezing a release, and the sealed test"],
   ["14", "What the app does with an upload"],
   ["15", "The decisions you can make, and what each one costs"],
   ["A–B", "File map and commands"]], [2.2, 14.2])
story.append(PageBreak())

toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-Bold", fontSize=10, leading=14, leftIndent=0),
                   ParagraphStyle("t1", fontName="Sans", fontSize=8.5, leading=11, leftIndent=16, textColor=MUTED)]
P("Contents", "h1")
story.append(toc)
story.append(PageBreak())

# 1
H1("1. The big picture")
P("The detector answers one question: <b>was this voice recording made by a person or by an AI voice generator?</b> "
  "It does not understand words. It looks for tiny statistical traces that generators leave in the sound — traces people usually cannot hear.")
P("Everything happens in two phases. In the <b>building phase</b> (on the training PC) we collect data, check it for traps, "
  "train the model, tune its scoring, and test it once. In the <b>using phase</b> (the app, on any laptop) the frozen result is applied to uploads.")
flow([
    ("1  Collect recordings", "ASVspoof 5, IndieFake (training) · In-the-Wild, VA (sealed tests) — detector/data/sources.py", colors.HexColor("#f1f5f9")),
    ("2  Describe and audit every file", "format, length, silence, duplicates, speaker splits, shortcut gates — build_manifest.py, audit.py, edge_probe.py", colors.HexColor("#f1f5f9")),
    ("3  Prepare the audio", "decode to 16 kHz mono, trim silence, cut edges, even loudness, fade — audio.py", colors.HexColor("#e0f2fe")),
    ("4  Augment (training only)", "random volume, noise, MP3/AAC/Opus/phone codecs, edge silence — augment.py", colors.HexColor("#e0f2fe")),
    ("5  Train the head on 4 s windows", "frozen XLS-R encoder + small trainable head, balanced batches — model/ssl.py, train/", colors.HexColor("#dcfce7")),
    ("6  Choose the best epoch on dev data", "dev error on clean + MP3 audio, early stopping — train/train.py", colors.HexColor("#dcfce7")),
    ("7  Combine 3 seeds and calibrate", "average logits, Platt scaling, verdict thresholds — calibrate.py", colors.HexColor("#fef9c3")),
    ("8  Freeze the release", "every file fingerprinted with SHA-256 — runs/release/v1", colors.HexColor("#fef9c3")),
    ("9  Sealed test, once", "In-the-Wild + VA, logged in a ledger — eval/evaluate.py", colors.HexColor("#fee2e2")),
    ("10  Serve in the app", "same preparation, every window scored, verdict + timeline — inference.py", colors.HexColor("#fee2e2")),
], box_h=29, gap=9, title="The whole pipeline in ten steps")
box("The model never sees file names, formats or labels at test time — only 4-second slices of cleaned-up sound. "
    "Most of the code exists to make sure that what it learns is the <i>voice</i>, not some accident of how the data was recorded or stored.")

# 2
H1("2. Words you will see everywhere")
T([["Word", "Meaning in this project"],
   ["Sample, 16 kHz", "Digital audio is a long list of numbers. 16 kHz = 16,000 numbers per second. Every file is converted to this, mono (one channel)."],
   ["Window", "A 4-second slice of audio = 64,000 numbers. The model always looks at exactly one window at a time."],
   ["Genuine / bona fide", "A real human recording (label 0)."],
   ["Fake / spoof", "Audio made by TTS (text-to-speech) or VC (voice conversion) (label 1). An <i>attack</i> is one particular generator (A01, A02, …)."],
   ["Logit", "The model's raw output score. Any number: very negative = sounds real, very positive = sounds fake."],
   ["Probability", "The logit turned into 0–100 % by <i>calibration</i> (section 11)."],
   ["Train / dev / test", "Train: what the model learns from. Dev: what we use to choose and tune (never learned from). Test: used once at the very end to report honest numbers."],
   ["Sealed test set", "A test set nobody looks at while building. In-the-Wild and VA are sealed; a ledger records every time they are run."],
   ["Speaker-disjoint", "No person appears in more than one of train/dev/test, so the model cannot pass by recognising voices."],
   ["Epoch", "One round of training: here 600 steps × 32 windows = 19,200 windows."],
   ["AUC", "How well scores separate real from fake. 1.0 = perfect, 0.5 = coin flip."],
   ["EER", "Equal error rate: the error when the threshold is set so 'real called fake' equals 'fake called real'. Lower is better."],
   ["Shortcut", "Something that predicts the label but is not about voices (file format, silence length). Models love shortcuts; they fail in the real world."],
   ["Encoder (XLS-R)", "A large pretrained speech network from Meta. We keep it frozen and use its internal 'understanding' of speech."],
   ["Head", "The small network we train on top of the encoder (183K parameters)."],
   ["Seed", "The starting random number. Training 3 seeds gives 3 slightly different heads; averaging them is more stable."]],
  [3.4, 13.0])

# 3
H1("3. Step 1 — The data")
P("Four collections are used. Each recording gets a <b>role</b>: train, dev or test. The roles are fixed before training and never change.")
T([["Source", "What it is", "Role", "Real", "Fake"],
   ["ASVspoof 5 (2024)", "International anti-spoofing challenge. Real speakers (audiobooks) vs. 8 generator systems in train. Official train and dev partitions.", "train", "7,267", "65,406"],
   ["", "", "dev", "10,652", "36,692"],
   ["IndieFake", "Indian speakers (English), real vs. fake. No official split, so we split by speaker (section 4).", "train", "1,415", "1,875"],
   ["", "", "dev", "1,603", "1,921"],
   ["In-the-Wild", "31,712 clips of celebrities and politicians from the internet. 54 speakers.", "test (sealed)", "19,901", "11,811"],
   ["VA set", "44 clips: 22 real, 22 TTS/VC fakes.", "test (sealed)", "22", "22"]],
  [2.7, 7.8, 2.1, 1.8, 2.0])
P("Counts are after the audit (section 4) removed bad, duplicate or unbalanced files.", "small")
H2("How a recording is found")
P("Datasets arrive as huge archives (tar or zip files). Unpacking them would double the disk use, so each recording is described by a "
  "<b>locator</b> — a short address the code can read the bytes from directly:")
CODE("""file:<path>                       an ordinary file
zip:<archive>::<member>           a file inside a zip (In-the-Wild)
tar:<archive>:<offset>:<size>     a byte range inside a tar (ASVspoof 5)""")
box("detector/data/sources.py — one small function per dataset (asvspoof5, indiefake, in_the_wild, va) lists its recordings with label, "
    "speaker group, attack name and role. read_bytes(locator) fetches the raw bytes.", "file")
H2("Describing every file (the manifest)")
P("Before anything is trained, every recording is decoded once and measured. One row per recording goes into a CSV file, the <b>manifest</b>:")
B("<b>Stored format</b>: container (wav/flac/mp3…), codec, sample rate, channels — used only to <i>check for leaks</i>, never given to the model.",
  "<b>Durations</b>: stored length and speech length.",
  "<b>Silence</b>: how much quiet there is at the start and at the end.",
  "<b>Levels</b>: peak loudness and average loudness of the speech.",
  "<b>Content fingerprint</b>: a SHA-256 hash of the decoded sound (pcm_hash), so the same audio stored twice is caught even if the file bytes differ.")
box("detector/data/build_manifest.py writes detector/runs/manifests/&lt;source&gt;.csv, using several CPU cores. It is incremental: re-running only processes new files.", "file")

# 4
H1("4. Step 2 — The audit: cleaning and checking the data")
P("The audit merges all manifests and applies rules learned the hard way from the <b>old (legacy) model</b>. That model scored well in the lab "
  "but failed in the real world, largely because the <i>file format</i> told it the answer. The audit makes such mistakes impossible to repeat silently.")
H2("Rule 1 — throw out unusable files")
P("Files that fail to decode, or have less than 0.5 s of speech, are excluded.")
H2("Rule 2 — duplicates")
P("If the same sound appears twice (same fingerprint): if the copies disagree on the label, all copies go; if any copy is in a test set, "
  "the train/dev copies go (otherwise the model would have 'seen the exam'); otherwise one copy is kept. 325 IndieFake duplicates and "
  "9 copies of test audio were removed.")
H2("Rule 3 — split by speaker")
P("IndieFake has no official split, so speakers are randomly assigned (seed 2026): 20 % of speaker groups to dev, the rest to train. "
  "1,040 IndieFake files whose speaker could not be identified were excluded — they cannot be safely placed. "
  "ASVspoof 5 already has speaker-disjoint official partitions.")
H2("Rule 4 — format balancing (the most important rule)")
P("Files are grouped into <b>format strata</b> — e.g. 'WAV, 16-bit, 44.1 kHz, mono'. If a stratum contains only fakes, the format itself "
  "is a perfect fake detector, and the model would learn the format instead of the voice. So, within train and within dev separately:")
B("strata missing either label are dropped entirely;",
  "the remaining strata are downsampled so they all have the <b>same fraction of fakes</b>.")
box("This is why 10,642 IndieFake files were excluded: most IndieFake fakes are stored in formats no real IndieFake file uses. Keeping them "
    "would have handed the model a shortcut. Less data, but honest data.", "warn")
H2("Rule 5 — the gates (training refuses to start if one fails)")
B("Every source must have <b>both</b> real and fake in train and in dev.",
  "Format must not predict the label: for every source, and for the pooled training mixture weighted exactly as the sampler draws it, "
  "the AUC of 'format → label' must be within 0.5 ± 0.10 (i.e. close to a coin flip).",
  "Length, silence and level shortcuts are measured and printed as warnings.")
P("The result is written to detector/runs/audit/manifest.csv and report.json with <b>passed: true/false</b>. The training code checks this "
  "flag and stops if the audit failed.")
box("detector/data/audit.py. Constants at the top: MIN_SPEECH_S = 0.5, DEV_FRACTION = 0.2, SPLIT_SEED = 2026, FORMAT_AUC_LIMIT = 0.60.", "file")

# 5
H1("5. Step 3 — The silence shortcut we found and removed")
P("During development the first models learned something surprising. In ASVspoof 5 training data:")
B("<b>Real recordings</b> have a little silence at the start and end (median 0.16 s and 0.22 s), and the voice <b>fades out</b> naturally.",
  "<b>Fakes</b> start and stop abruptly — no silence, full volume to the last sample.")
P("A simple test on just these edge features separated real from fake with <b>AUC 0.93</b> — without listening to the voice at all. "
  "A model would happily learn 'silence at the end = real', which is useless (and dangerous) in the real world.")
H2("How it was fixed")
B("<b>Trim with no margin</b>: silence at the edges is cut exactly where speech starts and ends.",
  "<b>Cut 0.25 s more from each edge</b> and <b>fade in and out over 0.15 s</b>: now every recording starts and ends the same way.",
  "<b>Training augmentation</b> randomly adds fades and very quiet padding to <i>both</i> classes, so edges carry no information.",
  "<b>Edge gate</b>: edge_probe.py re-measures the edge cues after preparation. Training is blocked if silence features exceed AUC 0.60.")
P("After the fix, edge cues are near chance (the tail-level AUC is 0.61, compared with 0.53 for the same measure in the middle of recordings — "
  "speech naturally gets quieter at sentence ends). A final check on dev data (edge_reliance.py) found only a mild remaining reliance on "
  "the first/last window (AUC 0.60–0.63).")
box("Shortcuts are the main reason detectors look great in papers and fail in apps. Finding and removing this one cost accuracy on "
    "ASVspoof 5 but made the In-the-Wild result trustworthy.", "warn")

# 6
H1("6. Step 4 — Preparing the audio (same everywhere)")
P("One function chain turns any file into model input. It is used <b>identically</b> in training, dev scoring, the sealed test and the app — "
  "a lesson from the legacy pipeline, where training and testing prepared audio differently.")
H2("6.1 Decode: any file → 16 kHz mono")
B("WAV/FLAC are read with <i>soundfile</i>; everything else (MP3, AAC, M4A, Opus, video files) with <i>ffmpeg</i>.",
  "Stereo is averaged to mono; other sample rates are resampled to 16,000 Hz with high-quality <i>soxr</i>.",
  "Values are clipped to −1…+1; broken files raise an error.")
H2("6.2 Prepare: trim, cut, level, fade")
prep_diagram()
T([["Step", "What happens", "Why"],
   ["Trim", "Measure loudness in 20 ms frames. Keep from the first to the last frame within 40 dB of the loudest frame (no margin).", "Silence length is a shortcut (section 5)."],
   ["Edge cut", "Remove 0.25 s from each end (or 10 % of the clip if shorter).", "Removes how a recording starts/stops."],
   ["Loudness", "Scale so the speech's average level is −25 dBFS; limit peaks to 0.99.", "Volume must not matter."],
   ["Fade", "Raised-cosine fade-in and fade-out of 0.15 s (at most ¼ of the clip).", "Every recording now has identical edges."]],
  [2.0, 9.0, 5.4])
H2("6.3 Windows: cut into 4-second pieces")
P("The model takes exactly 4 s. A recording is covered by <b>evenly spaced 4 s windows</b>; the number of windows is "
  "ceil(length ÷ 4 s), and the last one ends exactly at the end. A 10 s clip gets 3 windows (starting at 0, 3 and 6 s). Clips shorter "
  "than 4 s are <b>repeated</b> (looped) to fill the window — never padded with silence, because silence could become evidence.")
B("<b>Training</b>: one random 4 s crop per recording each time it is drawn.",
  "<b>Dev</b>: up to 4 windows per recording (covers up to 16 s).",
  "<b>Sealed test and app</b>: every window, however long the recording.")
P("<b>Minimum speech</b>: the app and test refuse to judge if less than <b>1.0 s</b> of speech remains after preparation "
  "(verdict 'not enough audio'). About 10 % of In-the-Wild clips are this short.")
box("detector/audio.py: decode(), prepare_with_offset(), window_starts(). Constants: SR = 16000, EDGE_CUT_S = 0.25, EDGE_FADE_S = 0.15. "
    "Changing anything here changes the model's input and requires retraining.", "file")

# 7
H1("7. Step 5 — Augmentation: making training audio messy on purpose")
P("Uploads are rarely studio-quality. They are compressed by WhatsApp (Opus), phones, MP3 or AAC, recorded at different volumes, with "
  "background noise. The legacy model flipped most real phone recordings to FAKE because it had never heard phone audio. "
  "So during training every window is randomly 'damaged' — <b>the same way for real and fake</b>, so damage itself carries no label.")
T([["Step (in order)", "Chance", "What it does"],
   ["Edge silence", "50 % + 50 %", "Random fades of 0.1–0.8 s at the ends; then very quiet noise (−75 to −50 dB) of up to 0.6 s added at the start and/or end."],
   ["Gain", "50 %", "Volume change between −8 dB and +4 dB."],
   ["Noise", "30 %", "White or 'pink-ish' noise at a signal-to-noise ratio of 10–40 dB."],
   ["Codec", "60 %", "Really encode and decode with ffmpeg, choosing one of 9 codecs at random (table below)."]],
  [3.2, 2.4, 10.8])
T([["Codec", "Imitates"],
   ["MP3 32k / 64k / 128k", "Downloaded and shared MP3s, from low to high quality"],
   ["AAC 48k / 96k", "iPhone voice memos, M4A files, Instagram/YouTube audio"],
   ["Opus 16k (VoIP) / 32k", "WhatsApp, Telegram, Discord voice notes"],
   ["µ-law 8 kHz", "Traditional phone line"],
   ["PCM 8 kHz", "Low-quality phone or old recorders"]],
  [5.0, 11.4])
box("Without augmentation the model would judge 'was this compressed?' instead of 'is this voice fake?'. Dev scoring also tests every "
    "model on MP3 32k audio, so a model that only works on clean audio is never chosen.", "note")
box("detector/augment.py: CODECS, augment(), edge_silence(), add_noise().", "file")

# 8
H1("8. Step 6 — The model")
model_diagram()
H2("8.1 The encoder: XLS-R 300M (frozen)")
P("XLS-R is a speech network published by Meta (wav2vec 2.0 family), pretrained <b>without labels</b> on about 436,000 hours of speech in 128 "
  "languages. It never learned about deepfakes; it learned what speech <i>is</i>. That general knowledge is what transfers to unseen audio.")
B("A <b>CNN front-end</b> turns the 64,000 numbers of a window into 199 frames, one every 20 ms.",
  "<b>24 transformer layers</b> refine those frames; each frame becomes 1,024 numbers. Lower layers describe sound texture (good for "
  "spotting artefacts); higher layers describe phonemes and words.",
  "<b>Frozen</b> means its 315.4 million parameters are never changed. Only the head learns. This makes training fast and stable on an "
  "8 GB GPU and avoids over-fitting to ASVspoof 5 — but it also limits how much the model can adapt (a decision point, section 15).")
H2("8.2 The head (trained, 182,939 parameters)")
T([["Part", "What it does", "Size"],
   ["Layer mix", "The encoder produces 25 outputs per frame (input + 24 layers). 25 learned weights (softmax) blend them into one. "
    "The trained models put the most weight on layers 4, 6 and 10.", "25"],
   ["LayerNorm", "Puts all features on a similar scale.", "2,048"],
   ["Projection", "Compress 1,024 → 128 numbers per frame (linear + GELU + 20 % dropout).", "131,200"],
   ["Attentive statistics pooling", "Learns which frames matter, then summarises the 199 frames into a weighted mean and weighted spread "
    "(128 + 128 = 256 numbers). Artefacts may appear only in some frames; attention finds them.", "16,641"],
   ["MLP classifier", "256 → 128 → 1 number: the <b>logit</b>.", "33,025"]],
  [3.4, 11.0, 2.0])
box("detector/model/ssl.py: SSLDetector and AttentiveStatsPool. The app uses an identical copy of the head (inference.py: Head) so it can "
    "share one encoder between the 3 trained heads; their outputs match training to 0.000000015.", "file")

# 9
H1("9. Step 7 — Training")
H2("9.1 Balanced batches")
P("ASVspoof 5 is about 90 % fake, IndieFake is about half and half. If recordings were drawn at random, the model would learn "
  "'ASVspoof-sounding = fake' — another shortcut. The <b>balanced cell sampler</b> first picks one of four cells with equal chance, "
  "then a random recording inside it:")
T([["Cell", "Recordings available", "Share of every batch"],
   ["ASVspoof 5 · real", "7,267", "25 %"], ["ASVspoof 5 · fake", "65,406", "25 %"],
   ["IndieFake · real", "1,415", "25 %"], ["IndieFake · fake", "1,875", "25 %"]],
  [5.0, 5.2, 6.2])
P("So the small IndieFake set is seen often (many times per epoch, with different random crops and damage each time), and real and "
  "fake are always 50/50.")
H2("9.2 One training step")
flow([
    ("Load 32 recordings", "decode → prepare → edge silence → random 4 s crop → augmentation  (6 CPU worker processes in parallel)", colors.HexColor("#f1f5f9")),
    ("Forward pass on the GPU", "frozen XLS-R → head → 32 logits  (bfloat16 to save memory)", colors.HexColor("#e0f2fe")),
    ("Loss", "binary cross-entropy: how wrong the logits are compared with the true labels", colors.HexColor("#dcfce7")),
    ("Update the head", "gradients clipped at 5.0; AdamW optimiser (weight decay 0.01); learning rate follows a one-cycle schedule up to 0.001", colors.HexColor("#dcfce7")),
], box_h=29, gap=9)
P("A recording that fails to load is marked and <b>left out of the loss</b> (it is not trained as silence). If more than 1 % fail in an "
  "epoch, training stops — the data is broken.")
H2("9.3 After every epoch: score dev and keep the best")
P("After each epoch (600 steps), the model scores a fixed dev subset — up to 400 recordings per (source, label, attack): "
  "<b>3,600 ASVspoof 5 + 800 IndieFake = 4,400 recordings</b> — twice: once clean and once after MP3 32 kbit/s compression.")
P("The <b>selection score</b> is the average of the four EERs (2 sources × 2 channels). If it is the best so far, the head is saved "
  "(best.pt). If 4 epochs in a row bring no improvement, training stops early. At most 12 epochs.")
T([["Final run", "Epochs run", "Best epoch", "Best selection EER"],
   ["Seed 42", "7", "3", "5.6 %"], ["Seed 43", "6", "2", "7.9 %"], ["Seed 44", "12", "10", "5.5 %"]],
  [4.2, 3.8, 3.8, 4.6])
P("Each epoch took about 3 min of training and 2.3 min of dev scoring on the RTX 5060. The early best epochs of seeds 42 and 43 suggest "
  "the head quickly starts to over-fit ASVspoof 5 — one reason the 3 heads are averaged.", "small")
box("detector/train/train.py (loop, selection) and detector/train/data.py (TrainWindows, BalancedCellSampler, DevWindows). "
    "Defaults: 12 epochs, 600 steps, batch 32, lr 1e-3, patience 4, dev 400 per cell. Each run writes config.json (with fingerprints of "
    "the exact data used), history.csv, best.pt, dev_predictions.csv and summary.json.", "file")

# 10
H1("10. Step 8 — Choosing the encoder and combining seeds")
P("Before the final training, two encoders were compared with one seed each on half the ASVspoof 5 training data. The rule was fixed "
  "<b>before</b> seeing results: pick the lower dev selection EER; a difference under 0.005 counts as a tie and goes to XLS-R "
  "(it is multilingual, which matters for Hindi).")
T([["Encoder", "Size", "Dev selection EER", "ASVspoof 5 clean / MP3", "IndieFake clean / MP3"],
   ["XLS-R 300M", "315M", "<b>5.1 %</b>", "1.0 % / 1.3 %", "8.4 % / 9.8 %"],
   ["WavLM Base+", "95M", "8.1 %", "3.1 % / 3.3 %", "12.5 % / 13.5 %"]],
  [3.0, 1.8, 3.2, 4.2, 4.2])
P("XLS-R won clearly. Three seeds (42, 43, 44) were then trained on the full data, and their logits are <b>averaged</b> — an "
  "<i>ensemble</i>. Different seeds make different small mistakes; averaging cancels some of them.")
box("detector/run_model_a.sh (comparison) and detector/run_final.sh (overnight chain: choose encoder → audit → 3 seeds → calibrate). "
    "detector/calibrate.py --compare prints the comparison table.", "file")

# 11
H1("11. Step 9 — Calibration: from score to probability and verdict")
P("A logit like −1.3 means nothing to a user. Calibration turns it into a probability and a verdict, using <b>dev data only</b>.")
H2("11.1 Building the calibration data")
B("Recording score = mean over its windows of (mean over the 3 heads).",
  "Clean and MP3-compressed dev scores are pooled (8,800 scores), so the thresholds also hold for compressed uploads.",
  "Every (source, label) cell gets equal total weight, so ASVspoof 5's 90 % fakes do not dominate. The resulting probability assumes a "
  "50/50 chance of real vs. fake before listening.")
H2("11.2 Platt scaling")
P("A logistic curve is fitted so that probabilities match how often audio at each score really is fake:")
CODE("probability = 1 / (1 + exp(−(0.6883 × logit + 1.7603)))\n\n"
     "logit −5 → 16 %    logit −3 → 42 %    logit −2.2 → 56 %    logit 0 → 85 %    logit +3 → 98 %")
H2("11.3 Thresholds from error budgets")
P("Two limits were chosen up front: at most <b>5 % of real audio</b> may be called 'likely fake', and at most <b>5 % of fakes</b> may be "
  "called 'likely real'. The code finds the thresholds that meet each budget on dev:")
T([["Probability", "Verdict"],
   ["below 0.514", "<font color='#15803d'><b>Likely real</b></font>"],
   ["0.514 – 0.562", "<font color='#b45309'><b>Uncertain</b></font>"],
   ["0.562 or more", "<font color='#b91c1c'><b>Likely fake</b></font>"],
   ["less than 1.0 s of speech", "<b>Not enough audio</b> (no score)"]],
  [6.0, 10.4])
box("The pooled dev error rate (5.4 %) is almost exactly the 5 % budgets, so the two thresholds land next to each other and the "
    "'uncertain' band is very narrow. Almost every clip gets a firm answer, and mistakes look confident. Tighter budgets (e.g. 2 %) "
    "would widen the band — a decision in section 15.", "warn")
T([["Dev slice", "n", "AUC", "EER", "Real → called fake", "Fake → caught"],
   ["Pooled (balanced)", "8,800", "0.990", "5.4 %", "5.0 %", "94.2 %"],
   ["ASVspoof 5 clean / MP3", "3,600 each", "1.000", "0.3 / 0.8 %", "0.0 %", "98 %"],
   ["IndieFake clean / MP3", "800 each", "0.97 / 0.96", "9.2 / 10.5 %", "≈ 10 %", "≈ 90 %"]],
  [3.9, 2.0, 2.0, 2.4, 3.1, 3.0])
box("detector/calibrate.py: ensemble(), cell_weights(), budget_thresholds(). FA_BUDGET and MISS_BUDGET are the two 5 % limits.", "file")

# 12
H1("12. Step 10 — Freezing the release")
P("Calibration writes detector/runs/release/v1/release.json: the encoder name, the 3 member runs with the SHA-256 fingerprint of each "
  "best.pt, the Platt numbers a and b, the thresholds, the dev results, and the fingerprint of the audit manifest. "
  "release.sha256 holds the fingerprint of release.json itself.")
B("The app and the test <b>refuse to load</b> a release whose files do not match their fingerprints.",
  "If anything changes — a retrained head, new thresholds — it is a <b>new release</b> (v2), with its own one-time test.",
  "This is why the laptop bundle must contain the exact files, and why check_install.py verifies their hashes.")

# 13
H1("13. Step 11 — The sealed test")
P("detector/eval/evaluate.py runs a frozen release on a sealed test set <b>once</b>. Before it starts it checks:")
B("the release fingerprints verify, and its members live inside the repository (no smoke-test releases);",
  "the audit manifest is exactly the one the release was calibrated against (so test membership cannot have changed);",
  "this (release, test set) pair is not already in the ledger (detector/runs/sealed_ledger.jsonl). A repeat needs --rerun and is logged.")
P("It then prepares every test recording exactly as the app would, scores every window, and reports AUC, EER, verdict rates, results "
  "by speech length and by attack, and <b>95 % confidence intervals</b> from 1,000 bootstrap resamples of whole speakers "
  "(people are the independent unit, not clips).")
H2("Results of release v1")
T([["Test set", "AUC [95 % CI]", "EER [95 % CI]", "Old model AUC / EER"],
   ["In-the-Wild (31,712)", "0.957 [0.941, 0.972]", "11.3 % [8.9, 13.5]", "0.839 / 23.4 %"],
   ["VA (44)", "0.963 [0.883, 1.000]", "6.8 % [0, 19.1]", "6 of 22 fakes caught"]],
  [4.0, 4.2, 4.2, 4.0])
T([["In-the-Wild", "Likely fake", "Uncertain", "Likely real", "Not enough audio"],
   ["Real (19,901)", "5.2 %", "0.5 %", "84.1 %", "10.2 %"],
   ["Fake (11,811)", "75.4 %", "0.9 %", "14.6 %", "9.1 %"]],
  [3.6, 3.2, 3.0, 3.2, 3.4])
T([["Speech in clip", "Clips", "AUC", "EER", "Fakes caught", "Real wrongly flagged"],
   ["under 4 s", "22,188", "0.934", "14.8 %", "59.5 %", "4.5 %"],
   ["4 – 16 s", "9,111", "0.992", "4.4 %", "98.3 %", "7.5 %"],
   ["over 16 s", "413", "0.999", "1.5 %", "98.9 %", "3.9 %"]],
  [3.0, 2.2, 2.2, 2.4, 3.0, 3.6])
box("Length matters most: with 4 s or more of speech the model catches about 98 % of fakes. The known weak spot is modern TTS "
    "(2025–2026 generators), which was not in the training data — both VA misses are TTS, and recent TTS samples made by the user pass as real.", "warn")

# 14
H1("14. Step 12 — What the app does with an upload")
flow([
    ("Upload", "browser sends the file to the FastAPI backend (app/); limits 50 MB and 10 minutes; nothing is stored on disk", colors.HexColor("#f1f5f9")),
    ("Decode + prepare", "exactly sections 6.1 and 6.2; remember where the prepared audio starts in the original", colors.HexColor("#e0f2fe")),
    ("Enough speech?", "under 1.0 s → 'not enough audio' and stop", colors.HexColor("#fef9c3")),
    ("Score every 4 s window", "encoder runs once per window; the 3 heads share it; mean of the 3 logits", colors.HexColor("#dcfce7")),
    ("Recording verdict", "mean window logit → Platt probability → thresholds → verdict", colors.HexColor("#dcfce7")),
    ("Timeline", "each window's own probability, placed at its real time in the upload (relative evidence, not a calibrated verdict)", colors.HexColor("#fee2e2")),
], box_h=29, gap=9)
P("On a GPU this takes well under a second per clip; on a desktop CPU a 28 s clip took about 2.6 s and gave the same verdicts "
  "(0.914 vs 0.916 on the same clip). A warning is shown for clips with under 4 s of speech.")
box("detector/inference.py: Detector (loads and verifies the release), analyze(data, filename) — the single function the app calls. "
    "Environment: DETECTOR_BACKEND=detector, DETECTOR_RELEASE (default runs/release/v1), DEVICE (auto/cpu/cuda).", "file")

# 15
H1("15. The decisions you can make")
P("Each row is a knob in the pipeline: what it is today, what changing it would do, and what it costs. Anything that changes the "
  "model's input, the model, or the thresholds needs a new release and a new one-time test.")
T([["Decision", "Today (v1)", "Options and trade-offs", "Effort"],
   ["Training data", "ASVspoof 5 + IndieFake (older generators)", "Add modern TTS (SpeechFake, MLAAD, IndicSynth + real Hindi, own generated fakes). "
    "<b>Biggest expected gain</b> for modern TTS. Must stay balanced per language/source.", "Days (+ downloads)"],
   ["Encoder frozen?", "Fully frozen; 183K trained", "Fine-tune the top layers or use LoRA: usually clearly better on real-world audio; "
    "slower training, more over-fitting risk, needs careful dev checks.", "2–3 days"],
   ["Encoder choice", "XLS-R 300M", "Larger (XLS-R 1B, w2v-BERT 2.0): maybe more accurate, slower, may not fit 8 GB. "
    "Smaller (WavLM Base+ 95M): 3× smaller, was worse on dev.", "1–3 days each"],
   ["Model size", "316M params, 1.2 GB", "Keep 3–6 layers + int8: 51–89 MB (accuracy loss unknown). "
    "Distillation into a ~50M student: ~50 MB, best accuracy per MB.", "1 day – 2 weeks"],
   ["Uncertain band", "0.514–0.562 (5 % budgets)", "2 % budgets → wider band, fewer confident mistakes, more 'uncertain' answers.", "Hours"],
   ["Calibration data", "Same dev used for selection", "Separate speaker split just for calibration/thresholds (more honest); per-length thresholds.", "1 day"],
   ["Minimum speech", "1.0 s", "Lower → fewer refusals but very unreliable answers on tiny clips; higher → more refusals.", "Hours"],
   ["Window length", "4 s", "Shorter windows help short clips; longer give more context. Needs retraining.", "1–2 days"],
   ["Augmentation", "9 codecs, noise, gain, edges", "Add Opus 16k stress tests, room echo, real phone recordings, RawBoost.", "1 day"],
   ["Test sets", "In-the-Wild, VA (sealed)", "Add VoxENES 2026, Deepfake-Eval-2024 and your own 50+50 set, all sealed.", "Download + 1 hour"]],
  [2.6, 3.3, 8.2, 2.3])
box("The usual order: (1) your own test set, (2) modern TTS training data, (3) fine-tuning, (4) wider uncertain band — all as release v2; "
    "then (5) shrink the v2 model for mobile. Shrinking v1 first would shrink a model that already misses new TTS.", "note")

# A
H1("A. File map")
T([["File", "What it does"],
   ["detector/paths.py", "Where datasets, runs and models live."],
   ["detector/audio.py", "Decode, prepare (trim, edge cut, loudness, fade), window schedule. Used everywhere."],
   ["detector/augment.py", "Training augmentation: codecs, noise, gain, edge silence."],
   ["detector/data/sources.py", "One adapter per dataset; read_bytes() for file/zip/tar locators."],
   ["detector/data/build_manifest.py", "Measure every file → runs/manifests/&lt;source&gt;.csv."],
   ["detector/data/audit.py", "Exclusions, duplicates, speaker splits, format balancing, gates → runs/audit/."],
   ["detector/data/edge_probe.py", "Edge-silence shortcut gate after preparation."],
   ["detector/model/ssl.py", "Encoder + head (SSLDetector), attentive pooling."],
   ["detector/train/data.py", "Training windows, balanced sampler, dev windows."],
   ["detector/train/train.py", "Training loop, dev scoring per epoch, early stopping, outputs."],
   ["detector/calibrate.py", "Compare runs; ensemble, Platt, thresholds; write and fingerprint a release."],
   ["detector/eval/evaluate.py", "One-time sealed test with ledger and bootstrap intervals."],
   ["detector/eval/edge_reliance.py", "Dev-only check of how much the first/last window matters."],
   ["detector/eval/legacy_baseline.py", "Scores the old models on In-the-Wild for comparison."],
   ["detector/inference.py", "The released detector used by the app and the sealed test."],
   ["detector/run_model_a.sh, run_final.sh", "Unattended chains: encoder comparison; final 3 seeds + calibration."],
   ["detector/watch.sh", "Live training dashboard in the terminal."],
   ["detector/check_install.py, make_bundle.sh", "Laptop setup: verify an install; pack the model files."],
   ["detector/RESULTS.md", "All v1 numbers."],
   ["app/", "The website: FastAPI backend + React frontend."]],
  [5.6, 10.8])

# B
H1("B. Commands (repository root, training PC)")
CODE("""# 1. describe the data, audit it, check the edge shortcut
venv/bin/python -m detector.data.build_manifest asvspoof5
venv/bin/python -m detector.data.audit
venv/bin/python -m detector.data.edge_probe

# 2. train one head (writes detector/runs/models/<name>/)
venv/bin/python -m detector.train.train --encoder wav2vec2-xls-r-300m --seed 42 \\
    --out detector/runs/models/final_wav2vec2-xls-r-300m_s42
bash detector/watch.sh                       # live progress

# 3. compare runs, then calibrate an ensemble into a release (dev only)
venv/bin/python -m detector.calibrate --compare detector/runs/models/*
venv/bin/python -m detector.calibrate --runs <run1> <run2> <run3> --out detector/runs/release/v2

# 4. one-time sealed test (logged in detector/runs/sealed_ledger.jsonl)
venv/bin/python -m detector.eval.evaluate --release detector/runs/release/v1 \\
    --source in_the_wild --out detector/runs/eval/v1_in_the_wild

# 5. the app
DETECTOR_BACKEND=detector venv/bin/python -m uvicorn app.backend.main:app --port 8000""")
P("Generated by college/make_pipeline_pdf.py. If the code changes, regenerate this document so it stays true.", "small")


# ---------------------------------------------------------------- build
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm, topMargin=2.0 * cm,
                         bottomMargin=2.0 * cm, title="How the Deepfake Detector Works", author="deepfake_project",
                         subject="Plain-language walkthrough of the detector pipeline")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self.decorate)])

    @staticmethod
    def decorate(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Sans", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.3 * cm, 1.2 * cm, "How the Deepfake Detector Works · release v1")
        canvas.drawRightString(A4[0] - 2.3 * cm, 1.2 * cm, f"{doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(2.3 * cm, 1.5 * cm, A4[0] - 2.3 * cm, 1.5 * cm)
        canvas.restoreState()

    def afterFlowable(self, f):
        level = getattr(f, "toc_level", None)
        if level is not None:
            self.notify("TOCEntry", (level, f.getPlainText(), self.page))


Doc(OUT).multiBuild(story)
print(f"Wrote {OUT}")
