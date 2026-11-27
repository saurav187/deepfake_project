"""Generate 4 clean, high-resolution (300 DPI) academic charts for the VoiceSleuth final report."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-voicesleuth")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/saurav/deepfake_project/report/assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#475569",
    "axes.linewidth": 0.8,
    "xtick.color": "#1e293b",
    "ytick.color": "#1e293b",
    "text.color": "#0f172a",
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cbd5e1",
    "figure.facecolor": "white",
})

C_V2 = "#1d4ed8"     # deep blue
C_V1C = "#d97706"    # amber
C_V1S = "#64748b"    # slate grey
C_GREEN = "#15803d"
C_RED = "#b91c1c"
C_PURPLE = "#7c3aed"

# ------------------------------------------------------------------ Figure 1: V2-90 vs Release v1 Detection
fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=300)
labels = ["Chatterbox\n(n=232)", "ElevenLabs-v3\n(n=120)", "FishTTS\n(n=283)",
          "Gemini-3.1\n(n=125)", "Hindi MLAAD\n(n=162)", "FreeVC24\n(n=150)"]
v2_vals = [100.0, 98.33, 100.0, 100.0, 99.38, 99.33]
v1_cal  = [78.02, 49.17, 99.29, 26.40, 82.72, 90.67]
v1_ship = [36.21, 9.17, 88.34, 3.20, 45.68, 70.67]

x = np.arange(len(labels))
w = 0.26
b1 = ax.bar(x - w, v2_vals, width=w, label="V2-90 C4 (cut 0.883)", color=C_V2, edgecolor="white")
b2 = ax.bar(x,     v1_cal,  width=w, label="Release v1 Calibrated (0.028)", color=C_V1C, edgecolor="white")
b3 = ax.bar(x + w, v1_ship, width=w, label="Release v1 Shipped (0.562)", color=C_V1S, edgecolor="white")

ax.axhline(90.0, color=C_GREEN, linestyle="--", linewidth=1.1, label="90% Per-Family Target")
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.3)

ax.set_ylabel("Fake Detection Rate (%)")
ax.set_title("Clean-Audio Fake Detection: V2-90 Ensemble vs. Release v1 (Identical Test Clips)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 114)
ax.grid(axis="y", linestyle=":", alpha=0.55)
ax.set_axisbelow(True)
ax.legend(loc="lower right", ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_model_comparison.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------ Figure 2: Genuine FA by Source
fig, ax = plt.subplots(figsize=(7.6, 3.7), dpi=300)
sources = [
    "VCTK (n=135)",
    "AMI Meetings (n=24)",
    "Hindi IndicVoices (n=1,226)",
    "Hindi Kathbath (n=72)",
    "IndieFake Genuine (n=98)",
    "EdAcc Conversational (n=177)",
    "LibriTTS Audiobooks (n=391)",
]
fa_v2  = [0.00, 4.17, 7.26, 11.11, 14.29, 17.51, 28.64]
fa_v1c = [1.48, 70.83, 37.93, 13.89, 40.82, 49.15, 5.63]

y = np.arange(len(sources))
h = 0.35
bars1 = ax.barh(y + h/2, fa_v2, height=h, label="V2-90 C4 (EN pooled: 19.15%, HI: 7.47%)", color=C_V2)
bars2 = ax.barh(y - h/2, fa_v1c, height=h, label="Release v1 Calibrated (EN pooled: 20.36%, HI: 36.59%)", color=C_V1C)
ax.axvline(15.0, color=C_RED, linestyle="--", linewidth=1.1, label="15% English FA Limit")

for bars in (bars1, bars2):
    for bar in bars:
        wval = bar.get_width()
        ax.text(wval + 0.9, bar.get_y() + bar.get_height()/2, f"{wval:.1f}%", va="center", ha="left", fontsize=7.6)

ax.set_xlabel("Genuine False-Alarm Rate (%)")
ax.set_title("Genuine False Alarms by Corpus (Clean, Matched Rule)", fontsize=10.5, pad=8)
ax.set_yticks(y)
ax.set_yticklabels(sources, fontsize=8.5)
ax.set_xlim(0, 82)
ax.grid(axis="x", linestyle=":", alpha=0.55)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7.6)
fig.tight_layout()
fig.savefig(OUT / "fig2_genuine_fa_by_source.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 3: 5-Channel Codec Robustness
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300)
channels = ["Clean", "G.711 µ-law\n(8 kHz)", "G.726 32k\n(Unseen 8 kHz)", "Opus 12k", "GSM\n(8 kHz)"]
xc = np.arange(len(channels))
pooled_t4 = [99.74, 100.0, 100.0, 99.87, 99.21]
min_t4    = [98.33, 100.0, 100.0, 99.17, 98.33]
en_fa     = [19.15, 21.09, 24.61, 25.82, 15.15]
hi_fa     = [7.47,  7.55,  8.63,  8.71,  4.16]

ax.plot(xc, pooled_t4, marker="o", linewidth=2.2, color=C_V2, label="Pooled 4-Target Detection (%)")
ax.plot(xc, min_t4,    marker="s", linewidth=1.8, linestyle="--", color=C_PURPLE, label="Worst Single-Target Detection (%)")
ax.plot(xc, en_fa,     marker="^", linewidth=2.0, color=C_RED, label="English Genuine FA (%)")
ax.plot(xc, hi_fa,     marker="d", linewidth=1.8, color=C_GREEN, label="Hindi Genuine FA (%)")

ax.axhline(90.0, color="#94a3b8", linestyle=":", linewidth=1.0)
ax.axhline(15.0, color=C_RED, linestyle=":", linewidth=1.0, alpha=0.6)
ax.text(2.0, 86.2, "90% target", fontsize=7.5, color="#475569", ha="center")
ax.text(2.0, 12.2, "15% EN FA cap", fontsize=7.5, color=C_RED, ha="center")

for i in range(len(xc)):
    ax.text(xc[i], min_t4[i] - 5.2, f"{min_t4[i]:.1f}%", ha="center", fontsize=7.5, color=C_PURPLE)
    ax.text(xc[i], en_fa[i] + 2.6, f"{en_fa[i]:.1f}%", ha="center", fontsize=7.5, color=C_RED)
    ax.text(xc[i], hi_fa[i] - 4.6, f"{hi_fa[i]:.1f}%", ha="center", fontsize=7.5, color=C_GREEN)

ax.set_ylabel("Rate (%)")
ax.set_title("V2-90 Performance Across Clean & Simulated Call Codecs (Cutoff 0.8830)", fontsize=10.3)
ax.set_xticks(xc)
ax.set_xticklabels(channels, fontsize=8.5)
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle=":", alpha=0.55)
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig3_codec_robustness.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ Figure 4: Post-Hoc Operating Points (2% / 5% / 10% / 15%)
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300)
fa_pts = [2, 5, 10, 15]
cb = [96.55, 99.14, 100.0, 100.0]
el = [85.83, 91.67, 97.50, 97.50]
ft = [99.65, 100.0, 100.0, 100.0]
gm = [96.00, 100.0, 100.0, 100.0]
gsm_pool = [86.05, 93.29, 98.16, 99.21]

ax.plot(fa_pts, ft, marker="o", linewidth=2.0, color="#0284c7", label="FishTTS (clean, EER 0.78%)")
ax.plot(fa_pts, cb, marker="s", linewidth=2.0, color=C_V2, label="Chatterbox (clean, EER 2.17%)")
ax.plot(fa_pts, gm, marker="^", linewidth=2.0, color=C_GREEN, label="Gemini-3.1 (clean, EER 2.47%)")
ax.plot(fa_pts, el, marker="D", linewidth=2.0, color=C_V1C, label="ElevenLabs-v3 (clean, EER 6.73%)")
ax.plot(fa_pts, gsm_pool, marker="x", linewidth=1.7, linestyle="--", color=C_RED, label="Pooled 4-Target on GSM (EER 5.62%)")

ax.axhline(90.0, color="#64748b", linestyle=":", linewidth=1.0)
ax.text(14.8, 90.6, "90% target", fontsize=7.6, color="#475569", ha="right")
ax.set_xlabel("English Genuine False-Alarm Budget (Post-Hoc Descriptive, %)")
ax.set_ylabel("Fake Detection Rate (%)")
ax.set_title("Post-Hoc Detection vs. English False-Alarm Budget (Descriptive Only)")
ax.set_xticks(fa_pts)
ax.set_xticklabels(["~2%\n(16/825)", "~5%\n(41/825)", "~10%\n(82/825)", "~15%\n(123/825)"], fontsize=8.5)
ax.set_ylim(75, 103)
ax.grid(True, linestyle=":", alpha=0.55)
ax.legend(loc="lower right", fontsize=7.8)
fig.tight_layout()
fig.savefig(OUT / "fig4_operating_points.png", dpi=300)
plt.close(fig)

print("Generated 4 charts in", OUT)
