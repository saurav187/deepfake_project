"""Build college/6_PROJECT_PLAN_FROM_LITERATURE.pdf: what to do and how, step by step, grounded in the downloaded papers.

Usage (repo root):  venv/bin/python college/make_project_plan_pdf.py     (needs: pip install reportlab)
Paper numbers (#n) refer to the file-name prefixes in papers/ (see college/5_LITERATURE_MATRIX.md).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdf_style import *  # noqa: F403  (styles, story and helpers)

OUT = Path(__file__).with_name("6_PROJECT_PLAN_FROM_LITERATURE.pdf")
G1, G2, G3, G4 = (colors.HexColor(c) for c in ("#f1f5f9", "#e0f2fe", "#dcfce7", "#fef9c3"))
G5 = colors.HexColor("#fee2e2")

# ---------------------------------------------------------------- cover
story.append(Spacer(1, 4 * cm))
P("Project Plan from the Literature", "title")
story.append(Spacer(1, 0.5 * cm))
P("What to do and how — problem statement, literature review, research design and data collection — "
  "for a real-world audio deepfake detector for Hindi and English", "subtitle")
story.append(Spacer(1, 0.35 * cm))
P("Based on 54 downloaded papers (2023–2027), the v1 pilot results and the Codex v2 research · November 2026", "subtitle")
story.append(Spacer(1, 1.6 * cm))
T([["The four steps your mentor asked for", "Where in this document", "Output to show your mentor"],
   ["1. Identify the problem statement", "Section 2", "One-page problem statement, objectives, research questions, scope"],
   ["2. Literature review", "Section 3", "Review chapter organised by themes + comparison table + gap table"],
   ["3. Formulate the research design", "Section 4", "Hypotheses, variables, experiments, metrics, validity controls"],
   ["4. Data collection", "Section 5", "Dataset inventory, own-recording protocol, consent form, data sheet"],
   ["Then: build, evaluate, report", "Sections 6–7", "Timeline, deliverables checklist"]], [4.6, 3.4, 8.4])
box("Paper numbers like #13 refer to the number at the start of each file name in your papers/ folder. "
    "The full classification of all 54 papers is in college/5_LITERATURE_MATRIX.md and in Appendix A.", "file")
story.append(PageBreak())

toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="Sans-Bold", fontSize=10, leading=14),
                   ParagraphStyle("t1", fontName="Sans", fontSize=8.5, leading=11, leftIndent=16, textColor=MUTED)]
P("Contents", "h1")
story.append(toc)
story.append(PageBreak())

# ---------------------------------------------------------------- 1
H1("1. Scope of the project")
P("A clear scope protects you from doing too much. Most of the 54 papers are wider (video, multimodal, quantum) or narrower "
  "(one lab dataset) than your project. This is the scope everything below is planned for:")
T([["In scope (must do)", "Stretch (only if time allows)", "Out of scope (say so explicitly)"],
   ["Audio only: a recording or audio/video file is uploaded",
    "Partial fakes: highlight which part of a recording is fake (timeline)", "Video / face deepfakes (#10, 19, 24, 33, 38, 39, 47, 49, 53)"],
   ["Speech made by modern TTS and voice cloning (2024–26 tools)", "Smaller model for mobile (~50–150 MB) (#48, #59)", "Live phone-call interception, real-time streaming"],
   ["English and Hindi; Indian accents", "More Indian languages", "Speaker verification / identifying who is speaking"],
   ["Real-world conditions: MP3/AAC/WhatsApp (Opus), phone audio, short clips", "Explainability views (#10)", "Watermarking (#31), quantum models (#47, #56)"],
   ["Calibrated answer: likely real / uncertain / likely fake / not enough audio", "", "Commercial product, legal-grade forensic proof"],
   ["Working web app on a laptop (already built)", "", ""]],
  [5.6, 5.0, 5.8])
box("Your project is not 'yet another classifier on ASVspoof 2019'. Its distinctive scope is <b>real-world, multilingual, "
    "modern-TTS detection with honest evaluation and a usable app</b>. Every step below is chosen to serve that scope.", "note")

H2("What is already done (and how to present it)")
P("You have built ahead of your mentor's order. That is fine — present it as the <b>pilot study</b> that sharpened the "
  "research questions, not as the finished project:")
T([["Already done", "How to present it"],
   ["v1 detector: frozen XLS-R + trained head, 3-seed ensemble", "Pilot / baseline system (Research design, section 4)"],
   ["Audit, speaker splits, silence-shortcut fix, sealed tests", "Methodology contribution: evaluation hygiene (section 4.5)"],
   ["In-the-Wild AUC 0.957, EER 11.3 % (old model 23.4 %)", "Pilot result that motivates RQ1–RQ3 (section 2)"],
   ["Weakness: modern TTS passes as real", "Evidence for the problem statement (section 2)"],
   ["Web app; laptop setup; PDF of the pipeline", "Prototype; later the deployment chapter"],
   ["Codex v2 research + dataset download (Tier A)", "Input to data collection (section 5)"]],
  [7.4, 9.0])

# ---------------------------------------------------------------- 2
H1("2. Step 1 — Identify the problem statement")
P("A good problem statement has four parts: <b>context</b> (why it matters), <b>gap</b> (what existing work fails to do — with "
  "evidence), <b>aim</b> (what you will do) and <b>scope</b>. The papers give strong evidence for each part.")
H2("2.1 Context (why it matters)")
B("Voice cloning and TTS now produce speech that people cannot tell apart from real voices; it is used for fraud, "
  "impersonation and misinformation (#45 voice-cloning fraud review, #6 and #50 reviews, #36 voice phishing).",
  "Deepfake research is dominated by detection techniques, but real-world deployment remains an open challenge (#6, #51).")
H2("2.2 The gaps — evidence from the papers")
T([["#", "Gap", "Evidence in your papers"],
   ["G1", "<b>Lab results do not transfer to real-world audio.</b>",
    "Most papers report near-perfect EER on ASVspoof 2019 LA (often under 1 %). On the unseen In-the-Wild set, errors are 6–30 %: "
    "#13 6.8 %, #15 7.1 %, #5 9.1 %, #18 10.2 %, #12 19.4 %, #11 26.1 %, #3 28.6 %; plain SSL baselines about 24–30 % (cited in #5). "
    "#3 reaches 1.98 % EER inside In-the-Wild but 28.55 % when trained on ASVspoof 2019 and tested on it."],
   ["G2", "<b>Modern (2024–26) generators are rarely tested.</b>",
    "Only #18 tests CosyVoice 2 (187 official samples) and #34 builds an OpenVoice set; #32 shows codec-based synthesis "
    "(CosyVoice2, Chatterbox, Mimi/EnCodec tokens) blurs the real/fake boundary."],
   ["G3", "<b>Compression and AI-processing cause false alarms.</b>",
    "#42: speech enhancement and neural codecs confuse detectors; #32: codec-resynthesised audio may be labelled either way; "
    "#9 (CLAD): manipulation attacks (noise, resampling, compression) break detectors unless trained for them."],
   ["G4", "<b>Indian languages, especially Hindi, are missing.</b>",
    "Language-specific detectors exist for Bangla (#17) and Arabic (#44); #52 states accent/language effects are 'still missing' "
    "from the literature. None of the 54 papers builds or tests a Hindi detector."],
   ["G5", "<b>Few usable, calibrated systems.</b>",
    "Only #44 (Arabic smartphone app) and #40 (real-time claim) target deployment; #48 and #59 study efficiency. "
    "No paper gives users a calibrated 'real / uncertain / fake' decision."],
   ["G6", "<b>Evaluation hygiene is weak.</b>",
    "Random in-domain splits inflate results (#15 0.03 %, #16 EER 0.029, #57 98.4 % accuracy, all within In-the-Wild). "
    "#36 uses silence/breathing as evidence, while our pilot showed silence at recording edges is a dataset shortcut."]],
  [1.0, 4.4, 11.0])
H2("2.3 Draft problem statement (edit in your own words)")
box("Audio deepfake detectors report near-perfect accuracy on laboratory benchmarks, yet their error rises to 6–30 % on "
    "real-world recordings, they are rarely tested against modern voice-cloning and text-to-speech systems, they raise false "
    "alarms on compressed or processed audio, and no detector has been developed or evaluated for Hindi. This project designs, "
    "evaluates and deploys an audio deepfake detector for uploaded Hindi and English recordings that generalises to unseen modern "
    "generators and real-world channels, and that gives calibrated, trustworthy decisions through a usable application.", "draft")
H2("2.4 Objectives")
B("<b>O1</b> Build a detector that generalises to modern TTS/voice-cloning systems not seen in training.",
  "<b>O2</b> Make it robust to real-world channels: MP3, AAC, WhatsApp Opus, phone audio and short clips.",
  "<b>O3</b> Extend it to Hindi and Indian-accented English with matched real and fake data.",
  "<b>O4</b> Give calibrated decisions with an explicit 'uncertain' outcome and controlled false-alarm rate.",
  "<b>O5</b> Evaluate honestly: speaker-disjoint splits, shortcut checks, sealed test sets, confidence intervals.",
  "<b>O6</b> Deploy it as a web application that runs on an ordinary laptop.")
H2("2.5 Research questions")
T([["RQ", "Question", "Answered by"],
   ["RQ1", "Does adding matched modern-TTS training data improve detection of <i>unseen</i> generators?", "Experiment E1"],
   ["RQ2", "Does adapting the speech encoder (LoRA / partial fine-tuning) generalise better than a frozen encoder?", "E2 (literature: #13, #55)"],
   ["RQ3", "Which training-time channel simulation reduces false alarms on compressed real audio?", "E3 (literature: #9, #42)"],
   ["RQ4", "How well does the detector work for Hindi, and what data is needed for it?", "E1 + Hindi slices of every test"],
   ["RQ5", "Can calibrated thresholds with an 'uncertain' band reduce confident mistakes without refusing too often?", "E4"]],
  [1.3, 11.0, 4.1])

# ---------------------------------------------------------------- 3
H1("3. Step 2 — Literature review")
H2("3.1 Clean up the paper set first")
B("<b>Delete duplicates</b>: files 25–30 are copies of 1–6 (60 files → 54 papers). Two different papers are both numbered 40.",
  "<b>Set aside 10 video/audio-visual papers</b> (#10, 19, 24, 33, 38, 39, 40-getPDF, 47, 49, 53): one paragraph in the "
  "introduction is enough; they are outside an audio-only scope.",
  "<b>Weight evidence</b>: journal papers with cross-dataset tests count more than short conference papers that report only "
  "accuracy on one dataset (#40, #43).",
  "<b>Use a reference manager</b> (Zotero is free): import all PDFs, tag each with its theme below, export citations to Word/LaTeX.")
H2("3.2 Read each paper with the same template")
P("Fill one row per paper in a spreadsheet. This turns 54 PDFs into a table you can write from:")
T([["Column", "What to write"],
   ["Problem", "What failure or need the paper addresses"],
   ["Method", "Features / encoder / classifier / loss (e.g. 'XLS-R fine-tuned + layer attention')"],
   ["Training data", "Which datasets, which languages"],
   ["Test data", "In-domain only, or cross-dataset (In-the-Wild, others)?"],
   ["Key result", "EER / AUC on each test set, with the condition"],
   ["Robustness / deployment", "Codecs, noise, short clips, speed, model size, app?"],
   ["Limitation", "What it did not test (be specific)"],
   ["Relevance to us", "Which RQ / gap it supports, and how"]],
  [4.0, 12.4])
H2("3.3 Structure of the review chapter (themes, not one paper per paragraph)")
T([["Section", "Content", "Core papers"],
   ["Background", "Deepfake speech generation (TTS, VC, codec LMs); threats; benchmark history", "#6, #50, #45, #51, #32"],
   ["Handcrafted features", "MFCC/LFCC/CQCC/GFCC + CNN/ResNet/ML; interpretable and light, weaker transfer", "#3, #11, #23, #46, #20, #48, #57, #58, #14"],
   ["End-to-end & graph models", "Raw waveform, graph attention, cross-modal views", "#2, #21, #41, #7, #18"],
   ["Self-supervised (SSL) models", "wav2vec2 / XLS-R / WavLM / HuBERT front-ends; frozen vs fine-tuned vs adapters; layer use", "#13, #55, #15, #5, #12, #16, #37, #22"],
   ["Robustness & real-world", "Codecs, AI-processing, manipulation attacks, splicing, cross-dataset tests", "#42, #32, #9, #4, #34, #36"],
   ["Languages & deployment", "Low-resource languages, apps, efficiency, quantisation", "#44, #17, #52, #48, #59"],
   ["Synthesis & gaps", "Comparison table (3.4) and gap table (2.2); how this project responds", "—"]],
  [3.3, 8.6, 4.5])
H2("3.4 The comparison table to put in your review")
T([["Method (paper)", "Front-end", "Trained on", "In-the-Wild EER"],
   ["Gaussian-attention adapters (#55)", "WavLM, adapters", "ASVspoof 2019 LA", "6.11 %"],
   ["Wave–spectrogram aggregation (#41)", "waveform + spectrogram", "ASVspoof 2019 LA", "6.58 %"],
   ["MSLA-XLS-R (#13)", "XLS-R, fine-tuned", "ASVspoof 2019 LA", "6.83 %"],
   ["LHCC (#15)", "XLS-R frozen + ~40M head", "ASVspoof 2019 LA", "7.1 %"],
   ["MWAADD (#5)", "XLS-R + emotion", "ASVspoof 2019 LA", "9.08 %"],
   ["SIGNL (#18)", "graph, 5 % labels", "CFAD", "10.16 %"],
   ["<b>This project, v1 pilot</b>", "<b>XLS-R, frozen; 3 heads</b>", "<b>ASVspoof 5 + IndieFake</b>", "<b>11.3 % [8.9–13.5]</b>"],
   ["NSF-PINN (#12)", "WavLM + physics losses", "ASVspoof 2019 LA", "19.4 %"],
   ["Multi-cepstral MLP (#3)", "MFCC+LFCC+CQCC+GFCC", "ASVspoof 2019 LA", "28.55 %"],
   ["Earlier legacy model (this project)", "CNN-LSTM from scratch", "ASVspoof 2019", "23.4 %"]],
  [5.0, 4.0, 3.9, 3.5])
box("Write the caveats next to the table: training data differ, v1 excludes clips with under 1 s of speech and reports "
    "speaker-clustered intervals, and in-domain In-the-Wild numbers (#15, #16, #57) are not comparable. "
    "The pattern: most systems that beat v1 <b>adapt the SSL encoder</b> (#13, #55), and #15 does it with a frozen encoder but a ~40M-parameter head (v1's is 183K). Both routes are tested in RQ2.", "warn")
H2("3.5 Critical observations to make (this is what makes a review good)")
B("<b>Benchmarks are aging.</b> Most papers still train and test on ASVspoof 2019 (generators from 2019). Only a few use "
  "ASVspoof 5 (#7, #21, #18). Modern generators are almost absent (G2).",
  "<b>Accuracy is the wrong headline.</b> Many papers report accuracy on balanced in-domain splits; EER, AUC and the "
  "detection rate at a fixed false-alarm rate are more meaningful for an app.",
  "<b>Shortcuts are under-examined.</b> #36 treats silence and breathing as evidence; a detector can learn dataset artefacts "
  "such as silence length or file format instead of voice traces. Your audit and edge gate address this.",
  "<b>SSL layer behaviour is consistent.</b> #13 (citing a layer-wise study) finds lower-to-middle layers most discriminative; "
  "your v1 learned the highest weights on layers 4, 6 and 10 — independent confirmation.",
  "<b>Efficiency is studied separately from generalisation.</b> #48 and #59 shrink models but test in-domain; nobody shows a "
  "small model that also generalises — a stretch contribution for you.")
H2("3.6 Papers to add (foundations missing from the folder)")
T([["Paper", "Why you need it"],
   ["Müller et al., Does Audio Deepfake Detection Generalize? (Interspeech 2022)", "Introduces In-the-Wild, your main test set"],
   ["Müller et al., Speech is Silver, Silence is Golden (ASVspoof 2021 workshop)", "The silence shortcut your pipeline guards against"],
   ["Wang et al., ASVspoof 5 (2024–25) + ASVspoof 2019/2021 summaries", "Your training data and the benchmark history"],
   ["Tak et al., wav2vec 2.0 + data augmentation for spoofing (Odyssey 2022)", "Standard XLS-R baseline"],
   ["Jung et al., AASIST (ICASSP 2022); Tak et al., RawBoost (ICASSP 2022)", "Standard back-end and augmentation"],
   ["Babu et al., XLS-R (Interspeech 2022)", "The encoder you use"],
   ["MLAAD, SpeechFake, IndicSynth, IndieFake, Deepfake-Eval-2024, VoxENES 2026", "Dataset papers for section 5"]],
  [9.6, 6.8])

# ---------------------------------------------------------------- 4
H1("4. Step 3 — Formulate the research design")
H2("4.1 Type of research")
P("<b>Quantitative, experimental design.</b> You change one factor at a time (training data, encoder adaptation, augmentation, "
  "calibration) while keeping everything else fixed, and measure the effect on held-out data. The v1 system is the "
  "<b>baseline</b>; every experiment is compared with it.")
flow([("Pilot (done)", "v1 baseline, audit, shortcut fix, sealed In-the-Wild/VA results", G1),
      ("Data", "add matched modern-TTS and Hindi data; build speaker- and generator-held-out splits", G2),
      ("Experiments E0–E4", "one factor at a time, decided on development data only", G3),
      ("Freeze v2", "best configuration, calibrated thresholds, fingerprinted release", G4),
      ("Sealed evaluation, once", "In-the-Wild, VA, VoxENES 2026, Deepfake-Eval-2024, own Hindi/English set", G5)],
     box_h=28, gap=8, title="Overall research design")
H2("4.2 Hypotheses")
T([["", "Hypothesis", "Literature support"],
   ["H1", "Matched modern-TTS data (MLAAD + M-AILABS, IndicSynth + Kathbath) raises detection of held-out generators at a fixed 2 % false-alarm rate.", "G1, G2; #18, #34"],
   ["H2", "Adapting the top encoder layers (LoRA or partial fine-tuning) beats the frozen encoder on unseen data.", "#13, #55 (best In-the-Wild results)"],
   ["H3", "Realistic channel augmentation (Opus/AAC/MP3, phone band, playback) lowers false alarms on compressed real audio without lowering recall.", "#9, #42, #32"],
   ["H4", "A separate calibration split with risk-based thresholds widens the 'uncertain' band and reduces confident errors.", "v1 pilot (band 0.514–0.562)"]],
  [1.0, 11.0, 4.4])
H2("4.3 Variables")
T([["Type", "Variables"],
   ["Independent (what you change)", "Training data mix · encoder frozen vs adapted · augmentation recipe · calibration/threshold method · (stretch) model size"],
   ["Dependent (what you measure)", "EER · AUC · detection rate at 2 % false alarms · false-alarm rate on real audio · 'uncertain' and 'not enough audio' rates · Brier score"],
   ["Controlled (kept fixed)", "Preprocessing (audio.py) · 4 s windows · speaker-disjoint splits · random seeds · number of training steps · test sets"],
   ["Grouping (reported separately)", "Language (Hindi / English) · speech length (&lt;4 s, 4–16 s, &gt;16 s) · channel · generator seen / unseen"]],
  [4.4, 12.0])
H2("4.4 Experiments (scoped for one RTX 5060 and a semester)")
P("This is a lighter version of the Codex plan (college/RESEARCH_V2.md): only the experiments needed to answer RQ1–RQ5.", "small")
T([["Exp.", "Comparison", "Decide on", "GPU time"],
   ["E0", "Re-score v1 on the new development splits; metadata-only check for shortcuts (format, duration, silence)", "Baseline numbers; no shortcut above AUC 0.60", "≈1 h"],
   ["E1", "v1 recipe with vs without the new matched data; one generator family held out (e.g. ElevenLabs-v3)", "Recall on the held-out family at 2 % false alarms (RQ1, RQ4)", "3–6 h"],
   ["E2", "Frozen encoder vs LoRA on the top 4 layers vs unfreezing the top 2 layers (best data from E1)", "Same metric; must not hurt Hindi or English (RQ2)", "5–12 h"],
   ["E3", "v1 augmentation vs realistic channel recipe (paired codecs, WhatsApp-style Opus 16k, phone band, playback)", "False alarms on compressed real audio at equal recall (RQ3)", "2–6 h"],
   ["E4", "Thresholds fitted on a separate speaker split; 2 % false-alarm and 5 % miss budgets; per-length thresholds", "Confident errors vs coverage (RQ5)", "&lt;1 h"],
   ["E5*", "Stretch: keep 3–6 encoder layers + int8 (51–89 MB); compare accuracy and laptop speed", "Size/speed vs accuracy", "1–3 h"]],
  [1.2, 7.4, 5.6, 2.2])
P("Each winning setting is repeated with 2 extra seeds before it is accepted. Total about 12–30 GPU-hours.", "small")
H2("4.5 Evaluation protocol and validity")
T([["Threat to validity", "Control (most are already built in v1)"],
   ["Same speaker in train and test", "Speaker-disjoint splits; parent recording and all its fake versions kept together"],
   ["Model learns a shortcut (format, silence, dataset)", "Format-balancing audit gate, edge-silence gate, metadata-only probe in E0"],
   ["Tuning on the test set", "Sealed test sets run once per release; ledger; thresholds from development data only"],
   ["Over-optimistic numbers", "Speaker-clustered bootstrap 95 % intervals; report failures and refusals"],
   ["Results only on lab data", "Cross-dataset tests: In-the-Wild, VoxENES 2026 (modern TTS), Deepfake-Eval-2024 (real 2024 deepfakes), own set"],
   ["Generator overlap", "Label each test generator as seen / related / unseen; report unseen separately"]],
  [5.2, 11.2])
H2("4.6 Ethics")
B("Record or clone a person's voice only with written consent; explain the purpose and how data is stored and deleted.",
  "Use datasets within their licences (several are non-commercial / academic only); keep licence notes with the data.",
  "Never store uploaded audio in the app (already the case); present results as an aid, not proof.")

# ---------------------------------------------------------------- 5
H1("5. Step 4 — Data collection")
H2("5.1 Public datasets (already planned and partly downloaded)")
P("Rule: <b>every fake source comes with a matched real source</b> (same language, similar recording conditions), otherwise the "
  "model learns 'this dataset = fake'.")
T([["Role", "Dataset", "Real / fake", "Size"],
   ["Training (have)", "ASVspoof 5 · IndieFake (Indian English)", "both", "local"],
   ["Training (Tier A)", "MLAAD English: ElevenLabs-v3, Gemini-3.1-Flash-TTS, Chatterbox, FishTTS", "fake", "1.4 GB"],
   ["", "M-AILABS US English (the real originals behind MLAAD)", "real", "8.0 GB"],
   ["", "IndicSynth Hindi (XTTS-v2, VITS, FreeVC24), 14 of 107 shards", "fake", "7.1 GB"],
   ["", "Kathbath Hindi (the real originals behind IndicSynth)", "real", "16.0 GB"],
   ["", "IndicVoices Hindi (spontaneous real speech, false-alarm control)", "real", "2.9 GB"],
   ["Sealed tests", "In-the-Wild · VA (have)", "both", "local"],
   ["", "VoxENES 2026 (Qwen3-TTS, CosyVoice 3, VibeVoice, Chatterbox, Seed-VC …)", "both", "19.8 GB"],
   ["", "Deepfake-Eval-2024 (real deepfakes from 2024, 42 languages; needs approval)", "both", "4.1 GB"],
   ["Optional", "SpeechFake multilingual + Common Voice real; LlamaPartialSpoof (partial fakes)", "both", "136 / 28 GB"]],
  [2.8, 9.2, 2.0, 2.4])
P("When the HDD is connected, run the inventory (DOWNLOAD_STATUS.py) and verify sizes and checksums before using anything.", "small")
H2("5.2 Your own dataset (the part only you can collect)")
P("This fills gaps G2 and G4 directly and becomes your most convincing test. Keep it <b>separate</b>: people in your test set "
  "must never appear in training.")
T([["Item", "Plan"],
   ["Volunteers", "Test set: 20–25 people not used anywhere else. Optional development set: another 20–30 people."],
   ["Consent", "Signed form: purpose, voice cloning permission, storage, deletion on request, no public release."],
   ["Real recordings", "Per person: 10–15 clips of 5–30 s; Hindi and English; read sentences + one spontaneous answer; "
    "phone microphone; quiet room and normal room."],
   ["Channels", "Send each real clip through WhatsApp (Opus) and keep the original; optionally play back and re-record."],
   ["Fake recordings", "Same sentences generated with 3–4 modern tools: e.g. Chatterbox (Hindi/English), XTTS-v2, F5-TTS, "
    "a commercial voice (free tier) — clone only consenting voices; also pass fakes through WhatsApp."],
   ["Size", "About 50 real + 50 fake for the private test set at minimum; more if time allows."],
   ["Metadata sheet", "clip_id, person_id, language, real/fake, generator + version + date, device, channel (original / WhatsApp), "
    "duration, sentence id, consent id."]],
  [3.0, 13.4])
box("Pass real and fake clips through the <b>same</b> channels (both through WhatsApp, both recorded on the same phones). "
    "Otherwise the model or the test can be fooled by the channel instead of the voice — the same lesson as the silence shortcut.", "warn")
H2("5.3 Storage and organisation")
B("HDD formatted ext4 on the PC; folders <i>development/</i> and <i>sealed/</i> kept apart (the download script already does this).",
  "One manifest row per recording, with source, licence, speaker, language, generator, label and role.",
  "Pack audio into large shard files for training (HDDs are slow at many small random reads).")

# ---------------------------------------------------------------- 6
H1("6. After data collection: build, evaluate, report")
T([["Week", "Work", "Output"],
   ["1", "Problem statement, objectives, RQs, scope (section 2)", "1–2 page document for mentor"],
   ["2–3", "Literature review: template table for 44 audio papers + foundations; write themed chapter", "Review chapter + comparison and gap tables"],
   ["3", "Research design (section 4)", "Design chapter: hypotheses, variables, experiments, metrics, ethics"],
   ["3–5", "Data: verify HDD, add matched sources to manifests, audit; record own dataset in parallel", "Data inventory, consent forms, metadata sheet"],
   ["5–6", "E0 + E1 (data effect)", "Results table vs v1"],
   ["6–7", "E2 (encoder adaptation) + E3 (channels)", "Ablation tables"],
   ["8", "E4 calibration; freeze v2; sealed evaluation once", "Final results with intervals; app updated to v2"],
   ["9", "(Stretch) E5 small model; demo preparation", "Size/speed table; demo"],
   ["10", "Write-up: results, discussion, limitations, future work", "Final report + presentation"]],
  [1.4, 9.4, 5.6])
P("Adjust the weeks to your semester; the order is what matters.", "small")

# ---------------------------------------------------------------- 7
H1("7. Checklist of deliverables")
T([["Step", "Deliverable", "Done?"],
   ["1", "Problem statement with context, gaps (with citations), aim and scope", "☐"],
   ["1", "Objectives O1–O6 and research questions RQ1–RQ5", "☐"],
   ["2", "Zotero library, duplicates removed, papers tagged by theme", "☐"],
   ["2", "Extraction spreadsheet (one row per paper)", "☐"],
   ["2", "Themed review chapter + In-the-Wild comparison table + gap table", "☐"],
   ["3", "Design chapter: hypotheses, variables, experiments E0–E4, metrics, validity, ethics", "☐"],
   ["4", "HDD inventory verified; matched real/fake pairs confirmed", "☐"],
   ["4", "Consent form, recording protocol, own dataset + metadata sheet", "☐"],
   ["5+", "Experiment results vs v1; frozen v2; one sealed evaluation", "☐"],
   ["5+", "Updated app, demo, final report", "☐"]],
  [1.4, 13.0, 2.0])
box("When you meet your mentor, bring: the problem statement (section 2.3), the gap table (2.2), the comparison table (3.4) and "
    "the experiment table (4.4). Those four pages show that the project follows the literature and has a clear plan.", "note")

# ---------------------------------------------------------------- A
H1("A. Paper index by theme")
T([["Theme", "Papers (file number) — H = core, M = supporting, L = background/exclude"],
   ["Reviews", "#6 H, #50 H, #45 M, #8 M, #52 M, #51 L"],
   ["SSL models", "#13 H, #55 H, #15 H, #5 H, #12 H, #41 H, #37 M, #16 M, #22 M, #35 M"],
   ["Robustness / real-world", "#42 H, #32 H, #9 H, #18 H, #3 H, #36 H (critical), #7 M, #4 M, #34 M"],
   ["Handcrafted + CNN/RNN", "#23 M, #11 M, #46 M, #1 M, #2 M, #21 M, #20 M, #14 M, #54 M, #58 M, #57 M (caution), #43 L, #40 L"],
   ["Languages, apps, efficiency", "#44 H, #17 H, #48 H, #59 H, #31 L–M, #56 L"],
   ["Video / multimodal (exclude)", "#10, #19, #24, #33, #38, #39, #40 (getPDF), #47, #49, #53"],
   ["Duplicates (delete)", "#25 = #1, #26 = #2, #27 = #3, #28 = #4, #29 = #5, #30 = #6"]],
  [4.2, 12.2])
P("Generated by college/make_project_plan_pdf.py. Details per paper: college/5_LITERATURE_MATRIX.md.", "small")

build(OUT, "Project Plan from the Literature", "What to do and how, grounded in the downloaded papers")
