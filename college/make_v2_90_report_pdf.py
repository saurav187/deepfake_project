"""Build college/32_V2_90_EVALUATION_REPORT.pdf: detailed V2-90 evaluation report,
Release v1 baseline comparison, simulated call codecs, Hindi results, and post-hoc
operating-point analysis.

Usage (from worktree or repo root):
    venv/bin/python college/make_v2_90_report_pdf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdf_style import *  # noqa: F403

OUT = Path(__file__).with_name("32_V2_90_EVALUATION_REPORT.pdf")
G_BLUE = colors.HexColor("#e0f2fe")
G_GREEN = colors.HexColor("#dcfce7")
G_AMBER = colors.HexColor("#fef3c7")
G_SLATE = colors.HexColor("#f1f5f9")
G_RED = colors.HexColor("#fee2e2")

# ------------------------------------------------------------------ Cover Page
story.append(Spacer(1, 2.6 * cm))
P("V2-90 Audio Deepfake Detection", "title")
P("Held-Out Evaluation & Release v1 Comparison Report", "title")
story.append(Spacer(1, 0.5 * cm))
P("Modern English TTS (Chatterbox, ElevenLabs-v3, FishTTS, Gemini-3.1-Flash-TTS), "
  "Simulated Call Codecs, and Hindi Speech under a Frozen Calibration Protocol", "subtitle")
story.append(Spacer(1, 0.35 * cm))
P("Protocol 28 (Amendments 1–6) · Code Commit a4a7068 · 27 November 2026 (IST)", "subtitle")
story.append(Spacer(1, 1.2 * cm))

T([["Metric / Milestone", "V2-90 C4 Ensemble (c2 + c3)", "Release v1 (Shipped / Calibrated)"],
   ["Primary Clean English Verdict", "<b>MISS</b> (missed on EN genuine FA by +35 clips)", "Reference baseline on identical 3,263 clips"],
   ["Target Modern TTS Detection (4/4)", "<b>98.3% – 100.0%</b> (758/760 = <b>99.74%</b> pooled)", "45.9% pooled (shipped) / 72.9% (calibrated)"],
   ["English Genuine False Alarms (≤15%)", "<b>19.15%</b> (158/825; LibriTTS alone: 112/391 = 28.6%)", "4.48% (shipped) / 20.36% (calibrated)"],
   ["Simulated Call Codecs (4 codecs)", "<b>98.3% – 100.0%</b> target det; 15.15% – 25.82% EN FA", "Not evaluated (clean only in baseline)"],
   ["Hindi Speech (Secondary, Same Cutoff)", "<b>99.4%</b> MLAAD HI, <b>99.3%</b> FreeVC24; <b>7.47%</b> HI FA", "45.7% MLAAD HI (5.55% FA) / 82.7% (36.59% FA)"],
   ["Post-Hoc Clean ~2% EN FA (Descriptive)", "<b>95.92%</b> pooled (CB 96.6%, EL 85.8%, FT 99.7%, GM 96.0%)", "Pooled clean AUC (C4): <b>0.9958</b>, EER <b>2.72%</b>"]],
  [5.2, 6.2, 5.0])

box("This report strictly separates <b>development metrics</b> (used for model selection and calibration), "
    "<b>primary held-out protocol outcomes</b> (frozen cutoff 0.8830, single signed pass), and "
    "<b>post-hoc descriptive analyses</b> (2% / 5% / 10% FA curves and ROC-AUC/EER). All numbers come from "
    "SHA-256-verified outputs bound to sign-off <font name='Mono'>c879b05ce98c0f1c</font>.", "note")
story.append(PageBreak())

# ------------------------------------------------------------------ TOC
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle("t0", fontName="Sans-Bold", fontSize=10, leading=14),
    ParagraphStyle("t1", fontName="Sans", fontSize=8.5, leading=11, leftIndent=16, textColor=MUTED),
]
P("Contents", "h1")
story.append(toc)
story.append(PageBreak())

# ------------------------------------------------------------------ 1. Executive Summary
H1("1. Executive Summary & Plain-Language Verdict")
P("The V2-90 experiment tested whether a fine-tuned self-supervised speech detector ensemble can achieve at least "
  "<b>90% detection separately</b> on four modern commercial and open TTS families — <b>Chatterbox</b>, "
  "<b>ElevenLabs-v3</b>, <b>FishTTS</b>, and <b>Gemini-3.1-Flash-TTS</b> — while keeping genuine English false "
  "alarms at or below <b>15%</b> using a single cutoff frozen before testing.")

P("Under the binding evaluation protocol (<b>college/28_V2_90_PROTOCOL.md</b>, amendments 1–6), the primary clean "
  "English outcome is a formal <b>MISS</b> — but the reason for the miss is highly diagnostic:")
B("<b>Fake detection exceeded the 90% goal by a wide margin across every target family:</b> Chatterbox 232/232 "
  "(<b>100.0%</b>), ElevenLabs-v3 118/120 (<b>98.33%</b>), FishTTS 283/283 (<b>100.0%</b>), and Gemini-3.1-Flash-TTS "
  "125/125 (<b>100.0%</b>) — <b>758/760 (99.74%)</b> pooled.",
  "<b>English genuine false alarms exceeded the 15% cap:</b> 158/825 (<b>19.15%</b>, 95% cluster bootstrap "
  "[14.14%, 24.43%]), missing the ≤123/825 budget by <b>+35 recordings</b>.",
  "<b>False alarms were concentrated in one corpus (LibriTTS audiobooks):</b> LibriTTS accounted for <b>112 of the "
  "158 English false alarms (28.64% of 391 clips)</b>, whereas VCTK had <b>0.00%</b> (0/135), AMI meetings had "
  "<b>4.17%</b> (1/24), IndieFake genuine had <b>14.29%</b> (14/98), and EdAcc had <b>17.51%</b> (31/177).",
  "<b>Simulated call compression did not break detection:</b> Across 8 kHz G.711 µ-law, unseen 8 kHz G.726 32 kbps, "
  "12 kbps Opus, and 8 kHz GSM, target fake detection stayed between <b>98.3% and 100.0%</b>. GSM came within 2 clips "
  "of passing English FA (125/825 = 15.15%).",
  "<b>Huge gain over Release v1 on the exact same clips:</b> Release v1 caught only <b>45.9%</b> of the four targets "
  "at its shipped cutoff (3.2% Gemini, 9.2% ElevenLabs-v3) and <b>72.9%</b> even when independently recalibrated to "
  "15% on the same calibration set (where Release v1's test English FA rose to 20.36% and Hindi FA to 36.59%).",
  "<b>Strong bilingual generalization to Hindi:</b> Using the identical English-calibrated cutoff with zero Hindi "
  "retuning, V2-90 flagged only <b>7.47%</b> of Hindi genuine recordings (97/1,298) while catching <b>99.4%</b> "
  "(161/162) of MLAAD Hindi TTS and <b>99.3%</b> (149/150) of IndicSynth FreeVC24 voice conversion.")

box("<b>Scope of this claim:</b> This benchmark evaluates <b>familiar generator families</b> on unseen en_UK texts, "
    "unseen parent utterances, and (for ElevenLabs-v3 and Gemini) unseen stock voices. It does <b>not</b> prove "
    "zero-shot generalization to brand-new unseen TTS families, and simulated ffmpeg codecs are not real phone calls.",
    "warn")

# ------------------------------------------------------------------ 2. Experimental Design & Pipeline
H1("2. Experimental Design, Data & Leakage Controls")
P("All data preparation, training, selection, calibration, sign-off, and single-pass evaluation ran under code "
  "commit <font name='Mono'>a4a7068</font> (<font name='Mono'>pipeline_v3/v2_90.py</font>, SHA-256 "
  "<font name='Mono'>ab92cc83…</font>) with immutable dataset fingerprint "
  "<font name='Mono'>f8eb0fe0eaf42ebe…</font>.")

flow([
    ("1. Staged Prep & Leakage Gate (v2-l4, 38 min)",
     "Decode 16 kHz mono, trim silence, centre 4 s crop, RMS norm. 0 leakage across all 8 split checks.", G_SLATE),
    ("2. Preflight & Advisory Shortcut Audit",
     "All train/dev/cal/test minima passed. Advisory audit flagged joint/tilt/ZCR cues (>0.80 AUC).", G_AMBER),
    ("3. Candidate Fine-Tuning (C2 on L4, C3 on A100)",
     "C2: XLS-R 300M (6,000 steps, bs 32). C3: AntiDeepfake XLS-R 1B (806 tensors, 4,500 steps, bs 16).", G_BLUE),
    ("4. Prespecified Dev Selection & Calibration (A100)",
     "C4 (c2+c3 z-mean) won dev tie-break (0.9822). Cutoff 0.8830 set on 634 EN cal genuine (95/634 = 14.98%).", G_BLUE),
    ("5. Cryptographic Sign-Off & Single-Pass Eval + Baseline",
     "signoff.json bound test shards + weights + cutoff before scoring 5 channels + Release v1 baseline.", G_GREEN),
], title="V2-90 End-to-End Execution Pipeline (26–27 Nov 2026 IST)")

H2("2.1 Dataset Composition & Verified Split Counts")
P("Per Protocol 28 Amendments 4–6, neurally restored or upsampled 'genuine' corpora (<b>LibriTTS-R</b> and "
  "<b>GLOBE v2</b>) and lineage-less <b>VoxENES</b> were excluded. Genuine splits were grouped strictly by speaker, "
  "conversation (EdAcc), or meeting-speaker connected component (AMI, deduplicating IHM/SDM mic siblings). Fakes "
  "from MLAAD (<font name='Mono'>30c3dec7</font>) split cleanly by <font name='Mono'>en_US</font> (train/dev) vs. "
  "<font name='Mono'>en_UK</font> (test), with stock-voice families split 50/50 by voice hash (half A train/dev, "
  "half B test). Every leakage counter in <font name='Mono'>prep.json</font> was <b>0</b>.")

T([["Split", "Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1", "EN Genuine (rec / spk)", "HI Genuine (rec / spk)"],
   ["Train", "424 (13 v)", "247 (15 v)", "447 (13 v)", "276 (17 v)", "5,254 / 276", "2,575 / 317"],
   ["Dev", "49 (12 v)", "26 (11 v)", "54 (13 v)", "31 (16 v)", "600 / 32", "1,459 / 122"],
   ["Calibration", "—", "—", "—", "—", "634 / 126", "618 / 122"],
   ["Held-Out Test", "232 (2 cl)", "120 (11 v)", "283 (2 cl)", "125 (13 v)", "825 / 150", "1,298 / 267"]],
  [2.4, 2.0, 2.2, 2.0, 2.0, 3.0, 2.8])

P("Training also included 6 related MLAAD variants (Chatterbox-Turbo, Resemble.ai, ElevenLabs-Turbo-v2.5, "
  "ElevenLabs-v2-Multilingual, Fish-S2-Pro, Openaudio-S1-Mini), 21 diverse MLAAD systems plus RVC, German "
  "ElevenLabs-v3 (half-A voices only), IndieFake, and Hindi MLAAD + IndicSynth. Label-symmetric training augmentation "
  "applied random low-pass filtering (4–7.8 kHz, p=0.3), codec round trips (µ-law, A-law, GSM, Opus 8/16/24k, MP3; "
  "p=0.4; <b>G.726 held out</b>), ±6 dB gain, and light noise (p=0.2).")

# ------------------------------------------------------------------ 3. Training, Selection & Calibration
H1("3. Model Training, Selection & Calibration (Development Only)")
H2("3.1 Candidate Architectures & Training Behaviour")
T([["Candidate", "Backbone & Freeze Policy", "Schedule / Hardware", "Dev History (min-target key / pooled)", "Saved Best Step"],
   ["<b>C2</b>", "Meta <font name='Mono'>wav2vec2-xls-r-300m</font><br/>CNN frozen, all 24 transformer layers trainable",
    "6,000 steps, bs 32, lr_enc 2e-5<br/>Colab L4 (83 min)",
    "Step 500: 1.0 / 0.9777 → Peak 1.0 / 0.9989 (steps 2500–5000); Final 1.0 / 0.9978",
    "Step 500<br/>(<font name='Mono'>3c180a32…</font>)"],
   ["<b>C3</b>", "NII <font name='Mono'>xls-r-1b-anti-deepfake</font><br/>Rev 60c96543 (806 tensors), lower 24/48 layers frozen",
    "4,500 steps, bs 16, lr_enc 1e-5<br/>Colab A100 (63 min)",
    "Step 500: 1.0 / 0.9565 → Peak 1.0 / 0.9844 (step 4000); Final 1.0 / 0.9833",
    "Step 500<br/>(<font name='Mono'>3c688d8a…</font>)"]],
  [1.7, 4.7, 3.6, 4.2, 2.2])

box("<b>Important checkpoint observation:</b> Because <font name='Mono'>v2_90.py</font> updates "
    "<font name='Mono'>best.pt</font> strictly on <font name='Mono'>m['key'] &gt; best['key']</font>, and both C2 and "
    "C3 reached <font name='Mono'>key = 1.0</font> on development at step 500, both candidates froze their "
    "<font name='Mono'>best.pt</font> weights at step 500 even though pooled dev detection rose further through "
    "steps 2500–5000. Per the frozen protocol, step-500 weights were evaluated unmodified.", "warn")

H2("3.2 Prespecified Selection & English Calibration")
T([["Option", "Dev Min-Target Key", "Dev Target Det (CB / EL / FT / GM)", "Dev Pooled Det", "Selected?"],
   ["C2 only", "1.0000", "49/49 · 26/26 · 54/54 · 31/31", "0.9777", "No"],
   ["C3 only", "1.0000", "49/49 · 26/26 · 54/54 · 31/31", "0.9565", "No"],
   ["<b>C4 (C2 + C3 z-mean)</b>", "<b>1.0000</b>", "<b>49/49 · 26/26 · 54/54 · 31/31</b>", "<b>0.9822</b>", "<b>YES (Primary)</b>"]],
  [3.6, 2.6, 4.8, 2.6, 2.8])

P("<b>Ensemble standardization:</b> Each candidate's logit is z-scored using its development-genuine distribution "
  "(C2: <font name='Mono'>mu = -4.6351, sd = 1.9971</font>; C3: <font name='Mono'>mu = -3.9402, sd = 1.2250</font>) "
  "and averaged equally. On the 634 clean English calibration genuine clips (126 speakers), the lowest cutoff flagging "
  "≤ floor(0.15 × 634) = 95 clips was <b>0.8829656848957964</b> (flagging <b>95/634 = 14.98%</b>):")
B("Calibration FA by source: <b>LibriTTS 78/354 (22.0%)</b>, EdAcc 16/103 (15.5%), AMI 1/22 (4.5%), "
  "VCTK 0/142 (0.0%), IndieFake 0/13 (0.0%). Already at calibration time, LibriTTS accounted for 82% of flagged clips.")

# ------------------------------------------------------------------ 4. Held-Out Evaluation
H1("4. Held-Out Test Results (Frozen Cutoff 0.8830)")
H2("4.1 Primary Clean English & Hindi Breakdown")
T([["Test Cell", "Flagged / N", "Rate", "Wilson 95% CI", "Clusters", "Cluster Boot 95% CI"],
   ["<b>Chatterbox</b> (target)", "232 / 232", "<b>100.00%</b>", "[98.37%, 100.0%]", "2", "[100.0%, 100.0%]"],
   ["<b>ElevenLabs-v3</b> (target)", "118 / 120", "<b>98.33%</b>", "[94.13%, 99.54%]", "11", "[95.59%, 100.0%]"],
   ["<b>FishTTS</b> (target)", "283 / 283", "<b>100.00%</b>", "[98.66%, 100.0%]", "2", "[100.0%, 100.0%]"],
   ["<b>Gemini-3.1-Flash-TTS</b> (target)", "125 / 125", "<b>100.00%</b>", "[97.02%, 100.0%]", "13", "[100.0%, 100.0%]"],
   ["<b>English Genuine (pooled)</b>", "<b>158 / 825</b>", "<b>19.15%</b>", "[16.61%, 21.98%]", "150", "[14.14%, 24.43%]"],
   ["  • genuine/libritts", "112 / 391", "28.64%", "[24.39%, 33.32%]", "71", "[19.60%, 38.21%]"],
   ["  • genuine/edacc", "31 / 177", "17.51%", "[12.62%, 23.79%]", "36", "[12.25%, 23.16%]"],
   ["  • genuine/indiefake", "14 / 98", "14.29%", "[8.70%, 22.56%]", "11", "[1.94%, 32.22%]"],
   ["  • genuine/ami", "1 / 24", "4.17%", "[0.74%, 20.24%]", "12", "[0.00%, 15.00%]"],
   ["  • genuine/vctk", "0 / 135", "0.00%", "[0.00%, 2.77%]", "20", "[0.00%, 0.00%]"],
   ["  • genuine_read_only (Libri+VCTK)", "112 / 526", "21.29%", "[18.01%, 24.99%]", "91", "[14.09%, 28.60%]"],
   ["IndieFake (fakes, descriptive)", "95 / 103", "92.23%", "[85.42%, 96.01%]", "12", "[81.65%, 100.0%]"],
   ["<b>Hindi Genuine (secondary)</b>", "<b>97 / 1,298</b>", "<b>7.47%</b>", "[6.16%, 9.03%]", "267", "[5.09%, 10.17%]"],
   ["  • genuine/indicvoices", "89 / 1,226", "7.26%", "[5.94%, 8.85%]", "257", "[4.89%, 9.93%]"],
   ["  • genuine/kathbath", "8 / 72", "11.11%", "[5.74%, 20.42%]", "10", "[1.61%, 24.24%]"],
   ["IndicSynth/freevc24 (Hindi VC)", "149 / 150", "99.33%", "[96.32%, 99.88%]", "15", "[98.00%, 100.0%]"],
   ["hi__IndicF5 (Hindi TTS)", "36 / 37", "97.30%", "[86.18%, 99.52%]", "36", "[91.67%, 100.0%]"],
   ["hi__OmniVoice / Veena / VoxCPM2 / Voxtral", "125 / 125", "100.00%", "[87.1%–90.4%, 100%]", "2–29", "[100.0%, 100.0%]"]],
  [4.8, 2.0, 1.7, 3.1, 1.5, 3.3])

P("<b>Speaker-level false-alarm skew on clean English genuine:</b> Out of 150 test English speakers, <b>92 speakers "
  "(61.3%) had zero false alarms</b>, while 26 speakers had ≥50% FA. Among the 15 speakers with the highest FA rates "
  "(80%–100%), 13 were LibriTTS audiobook readers (e.g., <font name='Mono'>libritts/4446</font> 8/8, "
  "<font name='Mono'>libritts/8461</font> 8/8, <font name='Mono'>libritts/5639</font> 6/6), plus "
  "<font name='Mono'>indiefake/Speaker-6</font> (6/6) and <font name='Mono'>ami/FIO084</font> (1/1).")

H2("4.2 Secondary Simulated Call Codec Channels (Identical Cutoff 0.8830)")
T([["Channel", "Verdict (FA gap)", "Chatterbox", "ElevenLabs", "FishTTS", "Gemini", "EN Gen FA", "HI Gen FA", "HI MLAAD", "FreeVC24"],
   ["<b>clean</b>", "MISS (+35)", "100.0%", "98.3%", "100.0%", "100.0%", "<b>19.15%</b>", "7.47%", "99.4%", "99.3%"],
   ["<b>mulaw8k</b> (8k)", "MISS (+51)", "100.0%", "100.0%", "100.0%", "100.0%", "<b>21.09%</b>", "7.55%", "100.0%", "99.3%"],
   ["<b>g726</b> (unseen)", "MISS (+80)", "100.0%", "100.0%", "100.0%", "100.0%", "<b>24.61%</b>", "8.63%", "99.4%", "98.7%"],
   ["<b>opus12</b> (12k)", "MISS (+90)", "100.0%", "99.2%", "100.0%", "100.0%", "<b>25.82%</b>", "8.71%", "99.4%", "100.0%"],
   ["<b>gsm</b> (8k)", "MISS (+2)", "98.7%", "98.3%", "99.7%", "100.0%", "<b>15.15%</b>", "4.16%", "99.4%", "97.3%"]],
  [2.3, 2.0, 1.5, 1.5, 1.4, 1.4, 1.7, 1.6, 1.5, 1.5])

box("All three 8 kHz telephone codecs (<font name='Mono'>mulaw8k</font>, unseen <font name='Mono'>g726</font>, and "
    "<font name='Mono'>gsm</font>) completely eliminate spectral content above 4 kHz, yet target detection stayed at "
    "<b>98.3%–100.0%</b>. Under GSM, English FA actually dropped to 15.15% (125/825, just 2 clips above the 15% cap) "
    "and Hindi FA dropped to 4.16% (54/1,298).", "note")

# ------------------------------------------------------------------ 5. Release v1 Comparison
H1("5. Apples-to-Apples Comparison with Release v1")
P("Release v1 (<font name='Mono'>detector/runs/release/v1</font>, SHA-256 <font name='Mono'>bdb7748c…</font>; "
  "frozen XLS-R 300M with 3 heads trained on ASVspoof 5 + IndieFake) was scored on the identical 634 English "
  "calibration clips and 3,263 held-out test clips (clean 4 s crops), both at its shipped threshold "
  "(<b>0.56235</b>) and recalibrated via the exact same ≤15% rule (<b>0.02785</b>).")

T([["Cell (Clean, Same 3,263 Clips)", "V2-90 C4 (cut 0.8830)", "Release v1 Shipped (0.56235)", "Release v1 Calibrated (0.02785)"],
   ["<b>Chatterbox</b> (n=232)", "<b>100.0%</b> (232/232)", "36.2% (84/232)", "78.0% (181/232)"],
   ["<b>ElevenLabs-v3</b> (n=120)", "<b>98.3%</b> (118/120)", "9.2% (11/120)", "49.2% (59/120)"],
   ["<b>FishTTS</b> (n=283)", "<b>100.0%</b> (283/283)", "88.3% (250/283)", "99.3% (281/283)"],
   ["<b>Gemini-3.1-Flash-TTS</b> (n=125)", "<b>100.0%</b> (125/125)", "3.2% (4/125)", "26.4% (33/125)"],
   ["<b>Pooled 4 Targets (n=760)</b>", "<b>99.7% (758/760)</b>", "<b>45.9% (349/760)</b>", "<b>72.9% (554/760)</b>"],
   ["<b>English Genuine FA (n=825)</b>", "<b>19.15%</b> (158/825)", "<b>4.48%</b> (37/825)", "<b>20.36%</b> (168/825)"],
   ["  • libritts / vctk", "28.6% / 0.0%", "0.3% / 0.0%", "5.6% / 1.5%"],
   ["  • edacc / ami / indiefake", "17.5% / 4.2% / 14.3%", "9.6% / 12.5% / 16.3%", "49.2% / 70.8% / 40.8%"],
   ["<b>Hindi Genuine FA (n=1,298)</b>", "<b>7.47%</b> (97/1,298)", "<b>5.55%</b> (72/1,298)", "<b>36.59%</b> (475/1,298)"],
   ["IndieFake fakes (n=103)", "92.2% (95/103)", "94.2% (97/103)", "98.1% (101/103)"],
   ["IndicSynth/freevc24 (n=150)", "<b>99.3%</b> (149/150)", "70.7% (106/150)", "90.7% (136/150)"],
   ["Hindi MLAAD 5 systems (n=162)", "<b>99.4%</b> (161/162)", "45.7% (74/162)", "82.7% (134/162)"]],
  [4.8, 3.8, 3.9, 3.9])

P("<b>Key comparison takeaways:</b> (1) At matched calibration rules, V2-90 beats Release v1 by <b>+26.8 percentage "
  "points</b> pooled on modern TTS (<b>+73.6 pts</b> on Gemini, <b>+49.1 pts</b> on ElevenLabs-v3, <b>+22.0 pts</b> "
  "on Chatterbox) while achieving <b>lower</b> English FA (19.15% vs. 20.36%) and <b>5× lower</b> Hindi FA (7.47% vs. "
  "36.59%). (2) The two models have opposite genuine-corpus failure profiles: Release v1 had near-zero LibriTTS FA "
  "(0.3%–5.6%) because it trained on ASVspoof 5 (MLS audiobooks) but spiked on conversational AMI (70.8%) and EdAcc "
  "(49.2%), whereas V2-90 handles AMI (4.2%) and Hindi (7.5%) well but over-flags LibriTTS (28.6%).")

# ------------------------------------------------------------------ 6. Post-Hoc 2% / 5% / 10% FA Analysis
H1("6. Post-Hoc Operating-Point & ROC-AUC Analysis (Descriptive Only)")
P("To answer what detection rate V2-90 achieves at stricter false-alarm budgets (<b>2%</b>, <b>5%</b>, <b>10%</b>, "
  "and <b>15%</b> English genuine FA), we evaluated the saved <font name='Mono'>eval-scores.jsonl</font> post-hoc. "
  "<b>These numbers are strictly descriptive</b> and do not change the primary frozen MISS verdict.")

T([["Channel", "Pooled AUC / EER", "At 1.94% EN FA (16/825)\nCB / EL / FT / GM (Pooled)", "At 4.97% EN FA (41/825)\nCB / EL / FT / GM (Pooled)", "At 9.94% EN FA (82/825)\nPooled (HI FA)"],
   ["<b>clean</b>", "<b>0.9958 / 2.72%</b>", "96.6 / 85.8 / 99.7 / 96.0 (<b>95.9%</b>)", "99.1 / 91.7 / 100 / 100 (<b>98.4%</b>)", "<b>99.6%</b> (4.01%)"],
   ["<b>mulaw8k</b>", "0.9939 / 3.66%", "86.6 / 84.2 / 99.3 / 91.2 (<b>91.7%</b>)", "95.3 / 93.3 / 99.7 / 99.2 (<b>97.2%</b>)", "<b>99.5%</b> (2.85%)"],
   ["<b>g726</b>", "0.9929 / 4.48%", "84.9 / 82.5 / 98.6 / 93.6 (<b>91.1%</b>)", "92.7 / 92.5 / 99.7 / 98.4 (<b>96.2%</b>)", "<b>99.0%</b> (2.70%)"],
   ["<b>opus12</b>", "0.9930 / 3.97%", "87.1 / 81.7 / 98.9 / 89.6 (<b>91.1%</b>)", "97.0 / 89.2 / 99.7 / 97.6 (<b>96.8%</b>)", "<b>98.8%</b> (3.47%)"],
   ["<b>gsm</b>", "0.9877 / 5.62%", "75.0 / 79.2 / 97.5 / 87.2 (<b>86.1%</b>)", "87.5 / 90.8 / 98.2 / 95.2 (<b>93.3%</b>)", "<b>98.2%</b> (2.54%)"]],
  [1.8, 2.5, 4.6, 4.6, 2.9])

P("<b>At ~2% clean English FA (cutoff +2.3415, 16/825 genuine flagged):</b> AMI, EdAcc, and VCTK all drop to "
  "<b>0.0% FA</b>, LibriTTS drops to <b>3.07%</b> (12/391), IndieFake genuine to <b>4.08%</b> (4/98), and Hindi "
  "genuine to <b>0.62%</b> (8/1,298). Three of four target families remain above 96% (FishTTS 99.7%, Chatterbox 96.6%, "
  "Gemini 96.0%), ElevenLabs-v3 is 85.8% (103/120), Hindi MLAAD is 97.5% (158/162), and FreeVC24 is 91.3%. "
  "Individual clean EERs vs. English genuine are: FishTTS <b>0.78%</b>, Chatterbox <b>2.17%</b>, Gemini <b>2.47%</b>, "
  "ElevenLabs-v3 <b>6.73%</b>, and IndieFake <b>10.49%</b>.")

# ------------------------------------------------------------------ 7. Shortcut Audit & Limitations
H1("7. Shortcut Audit, Limitations & Integrity")
H2("7.1 Advisory Shortcut Audit (Train → Dev Simple-Feature Probes)")
T([["Target Family", "RMS", "Rolloff95", "Upper >4k", "Tilt", "Speech Ratio", "ZCR", "Native Dur / SR", "Joint", "Red Flags (>0.80)"],
   ["Chatterbox", "0.538", "0.663", "0.591", "0.556", "0.702", "0.732", "0.593 / 0.709", "<b>0.822</b>", "joint"],
   ["ElevenLabs-v3", "0.616", "0.716", "0.777", "<b>0.822</b>", "0.686", "<b>0.873</b>", "0.510 / 0.709", "<b>0.890</b>", "tilt, zcr, joint"],
   ["FishTTS", "0.506", "0.560", "0.554", "0.502", "0.717", "0.761", "0.582 / 0.709", "<b>0.813</b>", "joint"],
   ["Gemini-3.1", "0.544", "0.577", "0.792", "<b>0.829</b>", "0.761", "<b>0.818</b>", "0.641 / 0.709", "<b>0.882</b>", "tilt, zcr, joint"],
   ["Pooled", "0.503", "0.543", "0.580", "0.557", "0.640", "0.599", "0.579 / 0.731", "0.638", "<i>none</i>"]],
  [2.3, 1.1, 1.4, 1.4, 1.2, 1.6, 1.2, 2.2, 1.3, 2.7])

H2("7.2 Key Limitations to Disclose Honestly")
B("<b>Familiar families only:</b> Training included en_US clips from all four target families; unseen-family "
  "generalization to future TTS architectures was not tested here.",
  "<b>Unmatched genuine vs. fake test panel:</b> M-AILABS en_UK genuine parents (Elizabeth Klett) were unavailable on "
  "Drive, so genuine FA is measured on separate corpora (LibriTTS, VCTK, AMI, EdAcc, IndieFake).",
  "<b>Low voice diversity in Chatterbox and FishTTS test cells (2 clusters each):</b> Their bootstrap intervals "
  "reflect unseen texts/parents rather than diverse speakers.",
  "<b>Calibration shift on LibriTTS:</b> Calibration FA (22.0% on LibriTTS) drifted to 28.6% on test LibriTTS, "
  "driving the entire English FA miss. Audiobook acoustic style overlaps with MLAAD read-speech synthesis.",
  "<b>Step-500 checkpoint freezing:</b> Strict <font name='Mono'>&gt;</font> selection on dev <font name='Mono'>key</font> "
  "kept step-500 weights rather than later checkpoints with higher dev pooled detection.",
  "<b>No real telephone calls & 4 s crop requirement:</b> Real PSTN/VoIP/WhatsApp audio and short clips (&lt;4 s) "
  "remain unmeasured in this protocol.")

# ------------------------------------------------------------------ 8. Compute & Next Steps
H1("8. Compute Accounting & Recommended Next Steps")
T([["Stage", "Hardware", "Time (IST)", "Elapsed", "Notes / Artifacts"],
   ["Data Prep + Audit", "v2-l4", "22:16 – 22:54", "38 min", "Fingerprint <font name='Mono'>f8eb0fe0…</font>, 0 leakage"],
   ["C3 Smoke (resume)", "v2-a100", "23:02 – 23:10", "8 min", "806 tensors verified; excluded from selection"],
   ["C2 Training (6k)", "v2-l4", "23:53 – 01:16", "83 min", "SHA-256 verified; L4 released at 01:26 IST"],
   ["C3 Training (4.5k)", "v2-a100", "00:08 – 01:12", "63 min", "Replacement A100 post-outage"],
   ["Select + Cal + Signoff", "v2-a100", "01:22 – 01:26", "4 min", "C4 selected; cutoff 0.8830; signoff bound"],
   ["5-Channel Evaluation", "v2-a100", "01:26 – 03:15", "108 min", "Serial CPU ffmpeg bottleneck (~13k calls/ch)"],
   ["Release v1 Baseline", "v2-a100", "03:15 – 03:17", "2.3 min", "3,897 clean clips; <font name='Mono'>baseline.json</font>"],
   ["<b>Total Post-Recovery</b>", "L4 + A100", "00:03 – 03:36", "<b>~3.5 h</b>", "<b>129.26 → 96.89 CU (~32.4 CU used)</b>"]],
  [3.4, 1.8, 2.6, 1.8, 6.8])

H2("Recommended Next Steps")
B("<b>1. Release v2-a100 immediately:</b> All outputs are SHA-256 verified on Drive and locally; no further GPU work "
  "is queued.",
  "<b>2. Fix calibration & checkpoint selection:</b> Tie-break <font name='Mono'>best.pt</font> by "
  "<font name='Mono'>(key, pooled, -EER)</font> instead of <font name='Mono'>key</font> alone, and address the "
  "LibriTTS domain FA spike via matched audiobook controls or domain-aware calibration before shipping.",
  "<b>3. Parallelize evaluation codecs:</b> Cache ffmpeg-processed channel arrays once per channel across CPU cores "
  "to shrink 5-channel evaluation from 108 min (~15 CU) to ~3 min (~0.4 CU).",
  "<b>4. Seal a fresh independent confirmation panel:</b> Before promoting any recalibrated or distilled student "
  "model into <font name='Mono'>app/</font>, evaluate once on a new sealed panel containing unseen modern TTS "
  "families and real phone/WhatsApp recordings.")

build(OUT, "V2-90 Evaluation & Release v1 Comparison Report", "Audio Deepfake Detection — V2-90 Results")
