"""Build report/VoiceSleuth_Final_Report.pdf matching report/Template.pdf and DU Semester VII guidelines.

Usage (from repo root):
    venv/bin/python report/build_final_report.py
"""
import sys
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parent
ASSETS = REPORT_DIR / "assets"
sys.path.insert(0, str(REPORT_DIR))

from reportlab.lib.units import cm
from reportlab.platypus import Image, NextPageTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle

from references_data import REFERENCES
from report_layout import (
    AcademicDoc,
    CONTENT_W,
    HorizontalRule,
    STYLES,
    add_bullets,
    add_chapter,
    add_figure,
    add_h2,
    add_h3,
    add_p,
    add_table,
    draw_architecture_diagram,
    draw_dfd_diagram,
    draw_sequence_diagram,
    make_toc,
)

TITLE = "VoiceSleuth: Speaker-Calibrated Modern TTS Deepfake Detection"
OUT_PDF = REPORT_DIR / "VoiceSleuth_Final_Report.pdf"


def build_report():
    story = []

    # ==================================================================
    # COVER PAGE (unnumbered, matching report/Template.pdf page 1)
    # ==================================================================
    story.append(Spacer(1, 0.7 * cm))
    add_p(story, TITLE, "cover_title")
    story.append(Spacer(1, 0.35 * cm))

    logo_path = ASSETS / "du_logo-000.png"
    if logo_path.exists():
        logo = Image(str(logo_path), width=4.4 * cm, height=4.3 * cm)
        logo.hAlign = "CENTER"
        story.append(logo)
    story.append(Spacer(1, 0.45 * cm))

    add_p(story, "Cluster Innovation Centre", "cover_inst")
    add_p(story, "University of Delhi", "cover_inst")
    story.append(Spacer(1, 0.9 * cm))

    add_p(story, "Submitted By:", "cover_body")
    story.append(Spacer(1, 0.15 * cm))
    add_p(story, "Saurav Kumar", "cover_bold")
    add_p(story, "Abhijeet Kumar", "cover_bold")
    add_p(story, "Rupesh Bhagat", "cover_bold")
    story.append(Spacer(1, 0.9 * cm))

    add_p(story, "August 2026 – November 2026", "cover_body")
    story.append(Spacer(1, 0.55 * cm))
    add_p(story, "Semester VII Project Report for:", "cover_body")
    add_p(story, "B.Tech (Information Technology and Mathematical Innovations)", "cover_bold")
    add_p(story, "Research Project Component (UGCF-2022)", "cover_body")
    story.append(Spacer(1, 0.85 * cm))

    add_p(story, "under the supervision of:", "cover_body")
    story.append(Spacer(1, 0.15 * cm))
    add_p(story, "Dr. Dharmbir Prasad", "cover_bold")
    add_p(story, "Associate Professor, Cluster Innovation Centre", "cover_body")
    story.append(Spacer(1, 0.45 * cm))
    story.append(HorizontalRule(width=CONTENT_W, thickness=0.9))

    # Switch to Roman numerals for Certificate, Acknowledgement, TOC
    story.append(NextPageTemplate("roman"))
    story.append(PageBreak())

    # ==================================================================
    # CERTIFICATE OF ORIGINALITY (page i)
    # ==================================================================
    add_p(story, "Certificate of Originality", "front_h1")
    story.append(Spacer(1, 0.3 * cm))
    add_p(
        story,
        f'The work presented in this report titled <i>"{TITLE}"</i> has been carried out by '
        "<b>Saurav Kumar</b>, <b>Abhijeet Kumar</b>, and <b>Rupesh Bhagat</b> as part of the "
        "Semester VII Research Project requirements for the degree of <b>B.Tech (Information "
        "Technology and Mathematical Innovations)</b> at the Cluster Innovation Centre, University "
        "of Delhi, during August 2026 – November 2026, under the guidance of our mentor, "
        "<b>Dr. Dharmbir Prasad</b>, Associate Professor, Cluster Innovation Centre.",
    )
    story.append(Spacer(1, 0.3 * cm))
    add_p(
        story,
        "We declare that this report reflects our own experimental work, software engineering, "
        "and literature analysis, and that all external datasets, pretrained checkpoints, and "
        "published papers used in this project have been properly cited. To the best of our "
        "knowledge, this work is free from plagiarism and data manipulation.",
    )
    story.append(Spacer(1, 2.6 * cm))

    sig_students = Table(
        [
            [
                Paragraph("Signature of Student", STYLES["sig_left"]),
                Paragraph("Signature of Student", STYLES["sig_left"]),
                Paragraph("Signature of Student", STYLES["sig_left"]),
            ],
            [Spacer(1, 1.1 * cm), Spacer(1, 1.1 * cm), Spacer(1, 1.1 * cm)],
            [
                Paragraph("(SAURAV KUMAR)", STYLES["sig_left"]),
                Paragraph("(ABHIJEET KUMAR)", STYLES["sig_left"]),
                Paragraph("(RUPESH BHAGAT)", STYLES["sig_left"]),
            ],
        ],
        colWidths=[CONTENT_W / 3] * 3,
    )
    sig_students.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(sig_students)
    story.append(Spacer(1, 2.2 * cm))

    sig_mentor = Table(
        [
            [Paragraph("", STYLES["sig_left"]), Paragraph("Signature of Mentor", STYLES["sig_right"])],
            [Spacer(1, 1.1 * cm), Spacer(1, 1.1 * cm)],
            [Paragraph("", STYLES["sig_left"]), Paragraph("(Dr. DHARMBIR PRASAD)", STYLES["sig_right"])],
        ],
        colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.55],
    )
    sig_mentor.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(sig_mentor)

    story.append(PageBreak())

    # ==================================================================
    # ACKNOWLEDGEMENT (page ii)
    # ==================================================================
    add_p(story, "Acknowledgement", "front_h1")
    story.append(Spacer(1, 0.3 * cm))
    add_p(
        story,
        "We would like to thank our mentor, <b>Dr. Dharmbir Prasad</b>, Associate Professor at the "
        "Cluster Innovation Centre, University of Delhi, for guiding us throughout our Semester VII "
        f'project, <i>"{TITLE}"</i>, from August to November 2026. His regular feedback pushed us to '
        "move beyond quick lab accuracy numbers and check whether our models actually hold up on "
        "unseen speakers, newer voice generators, and phone-style audio compression.",
    )
    add_p(
        story,
        "We are also thankful to the faculty and staff of the Cluster Innovation Centre for "
        "providing an open research environment in our B.Tech (IT & MI) programme where building real "
        "software and running honest experiments go hand in hand. When early tests exposed data "
        "leakage and false alarms on real audio, that support gave us the room to step back, fix "
        "our evaluation setup, and report the true numbers clearly.",
    )
    add_p(
        story,
        "Finally, we thank the open-source research community behind ASVspoof 5, MLAAD, "
        "In-the-Wild, LibriTTS, VCTK, AMI, EdAcc, Kathbath, IndicVoices, IndicSynth, Meta XLS-R, "
        "and the NII Yamagishi Lab for releasing the datasets and pretrained speech encoders that "
        "made this study possible.",
    )
    story.append(Spacer(1, 1.2 * cm))
    add_p(story, "Saurav Kumar, Abhijeet Kumar, and Rupesh Bhagat", "body_left")
    add_p(story, "Cluster Innovation Centre, University of Delhi", "body_left")

    story.append(PageBreak())

    # ==================================================================
    # TABLE OF CONTENTS (pages iii–v)
    # ==================================================================
    add_p(story, "Table of Contents", "front_h1")
    toc = make_toc()
    story.append(toc)

    # Switch to Arabic numbering starting at page 1 for ABSTRACT
    story.append(NextPageTemplate("arabic"))
    story.append(PageBreak())

    # ==================================================================
    # ABSTRACT (Arabic page 1)
    # ==================================================================
    add_chapter(story, "ABSTRACT", page_break=False)
    add_p(
        story,
        "Voice cloning tools can now copy a human voice from just a few seconds of speech. "
        "Scammers already use these tools in phone calls and WhatsApp voice notes to impersonate "
        "family members and colleagues. Most academic deepfake detectors still look strong on "
        "older lab benchmarks like ASVspoof 2019, yet break down on modern commercial text-to-speech "
        "(TTS) systems, phone compression, or Indian speech.",
    )
    add_p(
        story,
        "In this project, we built <b>VoiceSleuth</b>, an end-to-end audio deepfake detection "
        "system made of two linked parts: a multi-stage speech detection pipeline and a "
        "privacy-first web application where users upload an audio or video clip and get a clear "
        "verdict with a clickable segment timeline. Over the semester (August–November 2026), we "
        "developed and tested three generations of detectors. First, we audited a legacy CNN-LSTM "
        "baseline, fixed cross-speaker duplicates, and showed why it failed outside its training "
        "corpus (23.4% Equal Error Rate on In-the-Wild). Second, we trained Release v1 using a "
        "frozen XLS-R 300M speech encoder with codec augmentation and Platt calibration, cutting "
        "sealed In-the-Wild Equal Error Rate to 11.3% (4.4% on speech over 4 seconds). However, "
        "when we tested Release v1 on newer 2025–2026 generators, it caught only 3.2% of "
        "Gemini-3.1-Flash-TTS and 9.2% of ElevenLabs-v3 at its shipped cutoff.",
    )
    add_p(
        story,
        "To close that gap, we built our third pipeline (<b>V2-90</b>) under a strict leakage-free "
        "protocol with SHA-256 sign-off before testing. We fine-tuned two self-supervised models—"
        "Meta XLS-R 300M (C2) and NII AntiDeepfake XLS-R 1B (C3)—and combined them into an "
        "equal-weight z-score ensemble (C4) calibrated on 634 separate genuine English clips. On "
        "3,263 held-out test clips, C4 detected 100.0% of Chatterbox, 98.3% of ElevenLabs-v3, "
        "100.0% of FishTTS, and 100.0% of Gemini-3.1-Flash-TTS (758/760 pooled), and held "
        "98.3%–100.0% across four simulated phone codecs (G.711 µ-law, unseen G.726, Opus 12k, "
        "GSM) plus 99.4% on Hindi TTS at 7.47% Hindi false alarms. Formally, the clean run "
        "recorded a <b>MISS</b> on the 15% English false-alarm cap (19.15%, or 158/825), driven "
        "mainly by LibriTTS audiobooks (28.64%). Post-hoc analysis shows clean pooled AUC of "
        "0.9958 (EER 2.72%) and 95.9% pooled target detection at ~2% English false alarms. The "
        "FastAPI and React web app passes all automated tests and keeps zero copies of user audio.",
    )

    # ==================================================================
    # CHAPTER 1: INTRODUCTION
    # ==================================================================
    add_chapter(story, "1  INTRODUCTION")

    add_h2(story, "1.1  Background of the Project")
    add_p(
        story,
        "Synthetic speech used to sound flat and robotic, so most listeners could tell when a "
        "recording was computer-generated. Over the last two years, neural text-to-speech (TTS) "
        "and voice conversion (VC) tools have changed that completely. Systems such as "
        "ElevenLabs-v3, Gemini-3.1-Flash-TTS, FishTTS, and Chatterbox can generate natural tone, "
        "breathing, pauses, and accents in seconds. Some open models run on a single consumer GPU "
        "and clone a person's voice from a ten-second clip pulled off social media.",
    )
    add_p(
        story,
        "At the same time, audio deepfake detection research has struggled to keep up with real "
        "use. Most published models are trained and checked on ASVspoof 2019, a clean English "
        "benchmark built around older vocoders. When those same models face audio from newer "
        "generators, noisy rooms, WhatsApp compression, or non-English speakers, their error rates "
        "often jump from under 1% to more than 25% [3, 11].",
    )

    add_h2(story, "1.2  Motivation and Contemporary Relevance")
    add_p(
        story,
        "Voice cloning is no longer a lab demo. Real financial crimes already rely on cloned "
        "voices, including the 2019 UK CEO fraud ($243,000) [14, 39], the UAE bank transfer scam "
        "($35 million) [35, 40], and viral fake clips of political leaders [49]. Closer to home, "
        "everyday users in India frequently receive urgent WhatsApp voice notes or phone calls "
        "claiming to be a relative in trouble. Unlike fake images, people rarely have a quick, "
        "private way to check a suspicious voice clip.",
    )
    add_p(
        story,
        "We also saw two practical problems that motivated this work. First, if a detector keeps "
        "calling real human recordings fake (a high false-alarm rate), users stop trusting it. "
        "Second, uploading personal voice notes to an unknown cloud server raises serious privacy "
        "risks. A useful tool must therefore balance three things at once: high detection on "
        "modern TTS, honest calibration so uncertainty is shown instead of hidden, and strict "
        "privacy where uploaded audio is deleted right after analysis.",
    )

    add_h2(story, "1.3  Problem Statement")
    add_p(
        story,
        "Existing open audio deepfake detectors suffer from four linked failures that prevent "
        "reliable everyday use:",
    )
    add_bullets(story, [
        "<b>Generator lag:</b> Detectors trained on older corpora (such as ASVspoof 2019 or even "
        "ASVspoof 5) miss modern 2025–2026 commercial and neural-codec TTS systems like "
        "ElevenLabs-v3 and Gemini-3.1-Flash-TTS.",
        "<b>Hidden evaluation shortcuts and leakage:</b> Many published pipelines reuse speakers "
        "across train and test splits, keep silent openings or file-format cues that give away the "
        "label, or tune thresholds on test audio.",
        "<b>Compression and channel fragility:</b> Standard messaging and telephone codecs "
        "(Opus, G.711 µ-law, G.726, GSM) strip high frequencies and alter waveforms, causing "
        "clean-trained detectors to flip real speech to fake or miss synthetic speech.",
        "<b>Lack of Indian language coverage and usable software:</b> Almost no audio deepfake "
        "papers test Hindi speech or Indian-accented English, and very few provide a private, "
        "model-swappable web tool that shows which parts of a clip look synthetic.",
    ])

    add_h2(story, "1.4  Aims and Objectives of the Project")
    add_p(
        story,
        "The overall aim of VoiceSleuth is to design, rigorously test, and deliver a calibrated "
        "modern TTS detection pipeline paired with a private upload-and-analyze web application. "
        "We set five concrete objectives for Semester VII:",
    )
    add_bullets(story, [
        "<b>Objective 1 (Literature & Gap Mapping):</b> Review at least 50 recent peer-reviewed "
        "papers on audio deepfake detection to map current methods, generalization gaps, and "
        "evaluation flaws.",
        "<b>Objective 2 (Leakage-Free Multi-Generation Pipeline):</b> Audit our legacy CNN-LSTM "
        "baseline, build an improved self-supervised Release v1 baseline, and engineer a strict "
        "V2-90 training and evaluation pipeline with zero speaker/parent/voice leakage and "
        "cryptographic pre-test sign-off.",
        "<b>Objective 3 (Modern TTS Detection Under a Fixed Cutoff):</b> Train and compare "
        "fine-tuned self-supervised speech encoders (XLS-R 300M and AntiDeepfake XLS-R 1B) with "
        "the target of reaching at least 90% detection separately on Chatterbox, ElevenLabs-v3, "
        "FishTTS, and Gemini-3.1-Flash-TTS at ≤15% English genuine false alarms under one frozen "
        "calibration cutoff.",
        "<b>Objective 4 (Codec & Bilingual Stress Testing):</b> Measure how the exact same "
        "frozen cutoff holds up under four simulated call codecs (including unseen G.726) and on "
        "Hindi genuine and synthetic speech, benchmarked directly against Release v1 on the same "
        "test clips.",
        "<b>Objective 5 (Privacy-First Web Application):</b> Build and test a FastAPI + React web "
        "app behind a fixed detector contract so models can be swapped without changing app code, "
        "with zero audio storage and local browser waveform playback.",
    ])

    add_h2(story, "1.5  Research Questions and Hypotheses")
    add_p(story, "To guide our experiments objectively, we formulated five research questions:")
    add_bullets(story, [
        "<b>RQ1:</b> How accurately does a frozen self-supervised baseline trained on ASVspoof 5 "
        "and IndieFake (Release v1) detect modern 2025–2026 TTS families such as ElevenLabs-v3 "
        "and Gemini-3.1-Flash-TTS?",
        "<b>RQ2:</b> Does fine-tuning XLS-R 300M and AntiDeepfake XLS-R 1B with label-symmetric "
        "codec augmentation and combining them via z-score ensembling raise modern TTS detection "
        "above 90% per family while staying within a 15% genuine false-alarm budget?",
        "<b>RQ3:</b> Does a single threshold calibrated on clean English genuine speakers transfer "
        "reliably across unseen English genuine corpora and four simulated call codecs (including "
        "an unseen codec, G.726)?",
        "<b>RQ4:</b> Can an English-calibrated detector generalize to Hindi TTS and voice "
        "conversion without excessive false alarms on real Hindi speech?",
        "<b>RQ5:</b> Can a browser-and-server architecture support audio/video upload, timeline "
        "inspection, and model swapping while keeping zero uploaded audio on disk or in logs?",
    ])

    add_p(story, "<b>Hypotheses and Operational Definitions:</b>")
    add_table(
        story,
        [
            ["ID", "Hypothesis Statement", "Null Hypothesis (H0)", "Measurable Variables & Criteria"],
            [
                "H1",
                "Fine-tuned SSL ensemble (V2-90 C4) significantly beats frozen Release v1 on held-out modern TTS families.",
                "No detection gain over Release v1 at matched calibration rules.",
                "Per-family detection rate (%) on Chatterbox, ElevenLabs-v3, FishTTS, Gemini-3.1 (n=760), Wilson & cluster bootstrap 95% CIs.",
            ],
            [
                "H2",
                "One cutoff set on separate English calibration genuine clips (≤15% FA) holds English test FA ≤15%.",
                "Test English genuine FA exceeds 15% (>123/825) due to cross-corpus domain shift.",
                "Pooled and per-corpus English genuine FA (%) across LibriTTS, VCTK, AMI, EdAcc, IndieFake (n=825, 150 speakers).",
            ],
            [
                "H3",
                "Label-symmetric codec augmentation preserves ≥90% detection under 8 kHz phone codecs (including unseen G.726).",
                "Band-limiting to 8 kHz drops modern TTS detection below 90%.",
                "Detection rate and FA across mulaw8k, g726 (unseen), opus12, and gsm at identical frozen cutoff.",
            ],
            [
                "H4",
                "Multilingual XLS-R fine-tuning with Hindi training data yields ≥90% Hindi fake detection and ≤15% Hindi genuine FA.",
                "Hindi genuine FA >15% or Hindi fake detection <90% at the English cutoff.",
                "Hindi genuine FA (Kathbath + IndicVoices, n=1,298) and fake detection (5 MLAAD Hindi systems + FreeVC24).",
            ],
        ],
        [1.1, 4.9, 4.3, 5.6],
        "Table 1.1: Hypotheses, null statements, and operational definitions used in VoiceSleuth.",
    )

    add_h2(story, "1.6  Scope and Novelty of the Project")
    add_p(
        story,
        "<b>Scope:</b> VoiceSleuth focuses on offline analysis of uploaded audio and video clips "
        "(up to 50 MB and 10 minutes), targeting English and Hindi speech generated by modern TTS "
        "and voice conversion systems under clean and compressed channels. Out of scope for "
        "Semester VII are live phone-call wiretapping, speaker identity verification ('is this "
        "really person X?'), video face-deepfake analysis, user login accounts, and legal forensic "
        "proof. Also, the V2-90 held-out test measures <i>familiar generator families</i> on unseen "
        "texts, unseen parent utterances, and unseen ElevenLabs/Gemini stock voices; zero-shot "
        "testing on brand-new unseen TTS families and real PSTN phone recordings is left for "
        "Semester VIII.",
    )
    add_p(
        story,
        "<b>Novelty and Contribution:</b> Rather than chasing another 0.1% gain on ASVspoof 2019, "
        "our project combines four things rarely put together in undergraduate or even published "
        "work: (1) strict anti-leakage data hygiene—removing neurally restored 'genuine' corpora "
        "(LibriTTS-R, GLOBE v2) and verifying zero speaker/conversation/meeting overlap; "
        "(2) write-once cryptographic sign-off binding weights, shards, and cutoffs before any "
        "test score is computed; (3) direct side-by-side testing across four 2025–2026 TTS "
        "systems, four phone codecs, and Hindi; and (4) a complete privacy-by-default web app "
        "ready to host any detector through one clean interface.",
    )

    add_h2(story, "1.7  Organization of the Report")
    add_p(
        story,
        "The rest of this report follows our college project format. Chapter 2 reviews 54 recent "
        "papers and lists the main research gaps. Chapter 3 covers feasibility, our scientific "
        "methodology (following DU Pathway RM-B), and study limitations. Chapter 4 gives the "
        "Software Requirements Specification (SRS) for the web app and detector contract. Chapter 5 "
        "presents the system architecture, data flow, sequence diagram, and UI/API design. "
        "Chapter 6 explains how the models and app were implemented, including the real bugs we "
        "caught and fixed. Chapter 7 documents our testing strategy and automated test cases. "
        "Chapter 8 presents all experimental results across three model generations, simulated "
        "codecs, Hindi speech, Release v1 comparison, and post-hoc 2% FA analysis. Chapter 9 "
        "concludes with our Semester VIII plan, and Chapter 10 lists all 60 references.",
    )

    # ==================================================================
    # CHAPTER 2: LITERATURE REVIEW
    # ==================================================================
    add_chapter(story, "2  LITERATURE REVIEW")

    add_h2(story, "2.1  Introduction and Search Strategy")
    add_p(
        story,
        "Before choosing a model or dataset for V2, we carried out a structured literature review "
        "to avoid repeating mistakes already known in the field. We searched five major scholarly "
        "databases—<b>IEEE Xplore</b>, <b>ScienceDirect (Elsevier)</b>, <b>SpringerLink</b>, "
        "<b>ACM Digital Library</b>, and <b>Scopus / Google Scholar</b> (including ICASSP and "
        "Interspeech proceedings)—using combinations of the keywords: <i>'audio deepfake "
        "detection'</i>, <i>'speech anti-spoofing'</i>, <i>'self-supervised speech "
        "representations XLS-R WavLM'</i>, <i>'cross-domain generalization In-the-Wild'</i>, "
        "<i>'codec robustness'</i>, <i>'neural vocoder'</i>, and <i>'voice cloning fraud'</i>.",
    )
    add_p(
        story,
        "We collected 60 PDFs, removed 6 exact duplicates, and analyzed <b>54 distinct "
        "peer-reviewed papers</b> published between 2023 and 2027 (mostly 2025–2026). The corpus "
        "balances primary empirical research articles with secondary surveys: <b>37 audio-only "
        "detection papers</b> [1–5, 7, 9, 11–18, 20–23, 26, 28–31, 34, 36–39, 41, 43, 49–54], "
        "<b>11 multimodal or video deepfake papers</b> [10, 19, 24, 25, 27, 32, 33, 35, 42, 44, "
        "48], and <b>6 surveys, topic-modelling studies, or position papers</b> [6, 8, 40, 45–47]. "
        "Alongside these 54 papers, we consulted 6 foundational dataset and pretrained-model "
        "publications [55–60], bringing our total reference base to 60 sources.",
    )

    add_h2(story, "2.2  Study of Existing Detection Approaches")
    add_p(
        story,
        "Across the 37 audio-only papers, detection methods fall into three main families: "
        "handcrafted cepstral features, learnable raw-audio/spectral networks, and large "
        "self-supervised speech encoders (SSL).",
    )
    add_p(
        story,
        "<b>Handcrafted cepstral and spectral features:</b> Many studies extract MFCC, LFCC, "
        "CQCC, or GFCC features and pass them to classical classifiers or CNN/Conformer heads "
        "[3, 7, 11, 20, 23, 43, 52, 53]. Alfano et al. [3] combined four cepstral families under "
        "speaker-independent folds and got 0.30% EER within FakeAVCelebV2, but when trained on "
        "ASVspoof 2019 LA and tested on In-the-Wild (ITW), EER jumped to <b>28.55%</b>. Similarly, "
        "DAMRN [11] reached 0.10% EER on ASVspoof 2019 LA yet degraded to <b>26.14%</b> on ITW. "
        "Falcón-López et al. [43] showed that 32 MFCCs work well for compact edge models, while "
        "FakeSleuthNeXt [20] reached strong within-dataset scores using wavelet statistics without "
        "neural networks, though at heavy CPU feature-extraction cost (~7,700 s for 19.8 GB).",
    )
    add_p(
        story,
        "<b>Learnable front-ends and graph/waveform networks:</b> Several papers replace static "
        "log-Mel compression with trainable front-ends or raw-waveform models. Trainable PCEN with "
        "RCDLNet [1] uses only 1.26M parameters and cuts 2019 LA EER from 5.37% to 2.81%. SGC-ADD "
        "[2] simplifies graph attention to speed up raw-audio inference (~0.274 ms/sample). ARPNet "
        "[21] reverses short speech segments to expose phase artifacts (1.98% on 2021 LA) but "
        "still rises to 29.56% on ASVspoof 5. WaveSpec [36] fuses a wav2vec2 waveform branch with "
        "a spectrogram branch using a single-center loss and achieves 1.90% on 2021 DF and "
        "<b>6.58% on ITW</b>.",
    )
    add_p(
        story,
        "<b>Self-supervised speech encoders (SSL):</b> The strongest cross-domain transfer across "
        "the entire literature comes from large pretrained speech backbones—XLS-R [55], WavLM, "
        "HuBERT, and Whisper—combined with multi-layer aggregation rather than using only the "
        "final layer. Table 2.1 compares representative systems on in-domain (ASVspoof 2019 LA) "
        "versus out-of-domain (ASVspoof 2021 DF and In-the-Wild [57]) benchmarks.",
    )

    add_table(
        story,
        [
            ["Paper / System", "Backbone / Features", "Main Idea", "2019 LA EER", "2021 DF EER", "ITW EER"],
            ["DAMRN [11]", "MFCC+LFCC+GFCC", "Multi-scale dilated attention", "0.10%", "—", "26.14%"],
            ["Alfano et al. [3]", "4-cepstra + MLP", "Speaker-independent domain stability", "—", "—", "28.55%"],
            ["NSF-PINN [12]", "WavLM-Base+ (6/6)", "Source-filter physics + SAM", "4.49%", "12.38%", "19.40%"],
            ["SIGNL [18]", "wav2vec2 + graph", "Non-contrastive SSL, 5% labels", "—", "7.88%", "10.16%"],
            ["MWAADD [5]", "XLS-R + emotion", "Wavelet + LSTM attention", "0.18%", "2.55%", "9.10%"],
            ["LHCC [15]", "Frozen XLS-R", "Low- vs high-layer consistency", "—", "4.06%", "7.10%"],
            ["MSLA-XLS-R [13]", "XLS-R (conv frozen)", "Per-layer attention + weighted sum", "0.14%", "2.11%", "6.83%"],
            ["WaveSpec [36]", "wav2vec2 + spec", "Cross-modal + single-center loss", "—", "1.90%", "6.58%"],
            ["Wang et al. [50]", "WavLM-Large", "Gaussian-attention adapters", "0.42%", "3.56%", "6.11%"],
        ],
        [2.7, 2.7, 4.1, 2.1, 2.1, 2.2],
        "Table 2.1: In-domain vs. out-of-domain Equal Error Rate (EER, lower is better) across reviewed papers.",
    )

    add_h2(story, "2.3  Real-World Robustness, Codecs, and Shortcuts")
    add_p(
        story,
        "A key lesson from our reading is that real audio is rarely clean studio WAV. CLAD [9] "
        "tested standard detectors under simple edits that humans barely notice: false acceptance "
        "jumped to <b>36.69%</b> under volume changes, <b>31.23%</b> under fade, and <b>51.28%</b> "
        "under noise. Contrastive training brought those errors below 1.63%.",
    )
    add_p(
        story,
        "Even trickier is benign AI processing and neural codecs. An ICASSP 2026 study [26] showed "
        "that neural codecs like EnCodec, DAC, and Mimi are used both to transmit real calls and "
        "to generate modern TTS (such as CosyVoice 2). When genuine speech passes through a "
        "neural codec or denoiser, detectors often push it toward the fake region [37]. This "
        "finding directly shaped our own data preparation: we removed LibriTTS-R (restored by the "
        "Miipher neural vocoder) and GLOBE v2 (super-sampled/enhanced) from our genuine training "
        "pool so our model would not learn that vocoder traces mean 'real'.",
    )
    add_p(
        story,
        "Finally, Wang et al. [50] used SHAP explanations to warn that models often exploit "
        "silent regions at the start and end of clips as corpus shortcuts rather than inspecting "
        "voiced speech (contrast with BTS-E [30], which tries to model breathing and silence). "
        "That matches our own early findings and motivated trimming exterior silence plus running "
        "simple-feature shortcut audits before training.",
    )

    add_h2(story, "2.4  Language Coverage, Calibration, and On-Device Deployment")
    add_p(
        story,
        "Most audio deepfake papers test only English speech, a bias flagged by [40] and [47]. "
        "Within our 54-paper corpus, non-English work covered Chinese [7, 18, 21, 29], Bangla "
        "[17], and Arabic [39], with <b>zero papers studying Hindi or Indian-accented English</b> "
        "against modern TTS. Mallinson et al. [47] also point out that accent and dialect mismatch "
        "can cause unfair false alarms on certain speaker groups.",
    )
    add_p(
        story,
        "Equally serious is missing calibration. Almost every paper reports threshold-free EER or "
        "accuracy on class-imbalanced sets—[51] is a stark warning where 89.2% zero-shot accuracy "
        "hid a chance-level balanced accuracy of 49.9%. Only a handful of papers discuss "
        "calibration or likelihood-ratio scoring [1, 14, 27]. Meanwhile, real users need a fixed "
        "operating threshold with a known false-alarm rate, plus an honest 'Can't tell' zone when "
        "evidence is weak. On the deployment side, only a few works show working apps or demos "
        "(RealACaller on Arabic Android [39], Gradio/web demos [25, 53], or quantization studies "
        "[54]), and almost none measure end-to-end latency alongside privacy guarantees.",
    )

    add_h2(story, "2.5  Multimodal and Proactive Approaches")
    add_p(
        story,
        "Eleven papers in our corpus focus on video or audio-visual detection, including lip-sync "
        "consistency (ART-AVDF [24]), network-jitter alignment [19], late multimodal fusion "
        "[10, 27, 32, 33, 35], video-only forensics [44, 48], and hybrid quantum experiments "
        "[42, 51] (which showed no practical advantage over classical models). DeepShield [25] "
        "combines passive detection with proactive audio watermarking. While video cues help "
        "when faces are visible, voice notes and phone calls contain audio only, so our project "
        "focuses strictly on audio-only detection while still allowing users to upload MP4/MOV "
        "videos and analyze their audio track.",
    )

    add_h2(story, "2.6  Gaps in Existing Literature and Systems")
    add_p(
        story,
        "Synthesizing all 54 papers exposed six clear research gaps that directly shaped the "
        "design of VoiceSleuth (Table 2.2).",
    )
    add_table(
        story,
        [
            ["Gap ID", "Gap Found in Literature", "Evidence from Reviewed Papers", "How VoiceSleuth Addresses It"],
            [
                "G1",
                "Outdated generators in standard benchmarks",
                "Most papers test 2019/2021 vocoders; only [14, 18, 26, 28, 54] touch 2024–26 tools.",
                "Trains and evaluates on 2025–26 MLAAD systems: Chatterbox, ElevenLabs-v3, FishTTS, Gemini-3.1.",
            ],
            [
                "G2",
                "Evaluation leakage and acoustic shortcuts",
                "Within-corpus splits reported as robustness [16, 20, 52]; silence shortcuts [30, 50].",
                "Zero-leakage speaker/meeting/conversation splits, silence trimming, shortcut probes, SHA-256 sign-off.",
            ],
            [
                "G3",
                "Poor survival under phone/messaging codecs",
                "Manipulation and codec drops up to 33–51% [9, 26, 37].",
                "Label-symmetric training augmentation + held-out testing on µ-law, unseen G.726, Opus 12k, GSM.",
            ],
            [
                "G4",
                "Missing Hindi and Indian-accent coverage",
                "Zero Hindi papers in the 54-paper review; accent bias warned in [47].",
                "Includes IndieFake, EdAcc (~40 accents), Kathbath, IndicVoices, IndicSynth, and 5 Hindi MLAAD TTS.",
            ],
            [
                "G5",
                "Missing fixed-cutoff calibration & false-alarm reporting",
                "Papers report EER/accuracy without fixed-threshold FA rates [1, 14, 51].",
                "Sets one frozen cutoff on disjoint calibration speakers and reports raw counts + 95% cluster CIs.",
            ],
            [
                "G6",
                "Lack of private, model-swappable user tools",
                "Few offline/private demos; privacy under-studied [46].",
                "FastAPI + React app with streaming RAM uploads, zero audio storage, and pluggable detector API.",
            ],
        ],
        [1.2, 3.3, 5.2, 6.2],
        "Table 2.2: Six major literature gaps and how VoiceSleuth addresses each one.",
    )

    # ==================================================================
    # CHAPTER 3: SYSTEM ANALYSIS, FEASIBILITY AND RESEARCH METHODOLOGY
    # ==================================================================
    add_chapter(story, "3  SYSTEM ANALYSIS, FEASIBILITY AND RESEARCH METHODOLOGY")

    add_h2(story, "3.1  Introduction")
    add_p(
        story,
        "Following the University of Delhi Semester VII guidelines, this project adopts "
        "<b>Research Pathway RM (B): Scientific and Related Research</b>. In this chapter we "
        "assess whether the project is technically, economically, and operationally feasible, "
        "lay out our exact scientific materials, controls, and reproducibility rules, and state "
        "the known limitations up front.",
    )

    add_h2(story, "3.2  Feasibility Study")

    add_h3(story, "3.2.1  Technical Feasibility")
    add_p(
        story,
        "Building a modern audio detector from scratch would require huge compute, but "
        "fine-tuning open self-supervised speech models—Meta's 300M-parameter XLS-R [55] and "
        "NII Yamagishi Lab's 1B-parameter AntiDeepfake XLS-R [56]—fits comfortably on a single "
        "Colab L4 (24 GB VRAM) or A100 (40 GB VRAM) when mixed-precision bfloat16 training and "
        "partial layer freezing are used. On the software side, FastAPI, React 18, TypeScript, "
        "and system FFmpeg are mature, open-source tools that run reliably on both Linux laptops "
        "and standard cloud servers.",
    )

    add_h3(story, "3.2.2  Economic Feasibility")
    add_p(
        story,
        "We kept costs tightly controlled for an undergraduate budget. Every dataset used in "
        "training and evaluation (ASVspoof 5, MLAAD, In-the-Wild, LibriTTS, VCTK, AMI, EdAcc, "
        "Kathbath, IndicVoices, IndicSynth, IndieFake) is freely available for academic research "
        "with <b>zero paid API synthesis</b>. All heavy training and 5-channel evaluation ran on "
        "Google Colab Compute Units (CU): the full V2-90 recovery, dual-candidate training "
        "(C2 on L4 for 83 min, C3 on A100 for 63 min), 5-channel evaluation (108 min), and "
        "Release v1 baseline (2.3 min) consumed <b>33.7 CU total</b> (balance went from 129.26 CU "
        "to 95.55 CU), and both GPUs were released right after verification (0.00 CU/hr idle burn).",
    )

    add_h3(story, "3.2.3  Operational Feasibility")
    add_p(
        story,
        "The web app is designed for non-technical users: drag and drop a recording, wait a few "
        "seconds, and read a plain-English verdict card (<i>Likely AI-generated</i>, <i>Can't "
        "tell</i>, <i>No strong signs of AI</i>, or <i>Not enough speech</i>) alongside a "
        "color-coded timeline and permanent caveat. Developers can run the full stack locally "
        "in four commands using a mock backend (no GPU needed) and switch to Release v1 or "
        "future V3 checkpoints via a single environment variable (<font name='Mono'>DETECTOR_BACKEND</font>).",
    )

    add_h2(story, "3.3  Research Methodology (Scientific Pathway RM-B)")
    add_p(
        story,
        "To meet scientific research standards (materials, justification, controls, replicates, "
        "reproducibility, ethics, and statistical analysis), we wrote and froze our experimental "
        "protocol (<font name='Mono'>college/28_V2_90_PROTOCOL.md</font>, Amendments 1–6) before "
        "any V2-90 test recording was scored.",
    )

    add_h3(story, "3.3.1  Materials: Datasets and Strict Exclusion Rules")
    add_p(
        story,
        "<b>Synthetic speech (fakes):</b> We used MLAAD [58] at pinned revision "
        "<font name='Mono'>30c3dec7</font>. Our four primary target families are "
        "<b>Chatterbox</b>, <b>ElevenLabs-v3</b>, <b>FishTTS</b>, and "
        "<b>Gemini-3.1-Flash-TTS</b>. For richer training context, we added six related sibling "
        "folders (Chatterbox-Turbo, Resemble.ai, ElevenLabs-Turbo-v2.5, "
        "ElevenLabs-v2-Multilingual, Fish-S2-Pro, Openaudio-S1-Mini), 21 diverse MLAAD English "
        "systems plus RVC voice conversion, German ElevenLabs-v3 (train/dev only), IndieFake "
        "fakes, five Hindi MLAAD TTS systems (IndicF5, OmniVoice, Veena, VoxCPM2, Voxtral), and "
        "IndicSynth Hindi voice conversion (FreeVC24, XTTS-v2, VITS).",
    )
    add_p(
        story,
        "<b>Genuine speech (real humans):</b> Original unprocessed <b>LibriTTS</b> [60], "
        "<b>VCTK</b> studio English, <b>AMI</b> meeting recordings (headset IHM and distant mic "
        "SDM), <b>EdAcc</b> conversational English across ~40 global accents, <b>M-AILABS "
        "en_US</b> (train/dev only), <b>IndieFake</b> Indian-accented English genuine clips, "
        "plus <b>Kathbath</b> and <b>IndicVoices</b> for Hindi genuine speech.",
    )
    add_p(
        story,
        "<b>Deliberate exclusions (quality control):</b> We dropped <b>LibriTTS-R</b> (restored "
        "with the Miipher neural vocoder) and <b>GLOBE v2</b> (enhanced and upsampled to "
        "44.1 kHz) because neural processing leaves synthetic traces on real labels. We also "
        "excluded <b>VoxENES</b> from the primary run because it lacks source/target speaker IDs "
        "needed to prove disjoint splits.",
    )

    add_h3(story, "3.3.2  Canonical Preprocessing, Split Rules, and Leakage Controls")
    add_p(
        story,
        "Every clip—real or fake, train or test—passes through the exact same preprocessing "
        "function: decode to 16 kHz mono, trim exterior low-energy silence, apply RMS loudness "
        "normalization, and take a <b>4-second crop</b> (random during training, centered for "
        "calibration and testing; clips shorter than 4 seconds after trimming are excluded and "
        "counted). One recording counts once per condition.",
    )
    add_p(
        story,
        "Data splits are fixed deterministically by SHA-256 hash before training begins: "
        "MLAAD <font name='Mono'>en_US</font> originals go to train/dev while "
        "<font name='Mono'>en_UK</font> originals go exclusively to test; ElevenLabs and Gemini "
        "stock voices are split 50/50 (half A train/dev, half B test); AMI is grouped by "
        "<b>connected components of meetings and speakers</b> (so no speaker or shared meeting "
        "room ever spans two splits, and IHM/SDM mic siblings are deduplicated); EdAcc is split "
        "by conversation ID; and LibriTTS, VCTK, IndieFake, Kathbath/IndicSynth, and IndicVoices "
        "are split strictly by speaker. Our automated leakage gate (<font name='Mono'>leakage()</font> "
        "in <font name='Mono'>v2_90.py</font>) aborts preparation if any of 8 leakage counters is "
        "non-zero.",
    )

    add_table(
        story,
        [
            ["Split", "Chatterbox", "ElevenLabs-v3", "FishTTS", "Gemini-3.1", "EN Genuine (rec / spk)", "HI Genuine (rec / spk)"],
            ["Train", "424 (13 v)", "247 (15 v)", "447 (13 v)", "276 (17 v)", "5,254 / 276", "2,575 / 317"],
            ["Dev", "49 (12 v)", "26 (11 v)", "54 (13 v)", "31 (16 v)", "600 / 32", "1,459 / 122"],
            ["Calibration", "—", "—", "—", "—", "634 / 126", "618 / 122"],
            ["Held-Out Test", "232 (2 cl)", "120 (11 v)", "283 (2 cl)", "125 (13 v)", "825 / 150", "1,298 / 267"],
        ],
        [2.3, 2.0, 2.2, 2.0, 2.0, 2.8, 2.6],
        "Table 3.1: Verified V2-90 dataset counts after 4 s cropping and zero-leakage gating (3,263 total test clips).",
    )

    add_h3(story, "3.3.3  Label-Symmetric Augmentation, Reproducibility, and Statistics")
    add_p(
        story,
        "During training, both genuine and synthetic clips receive the exact same augmentation "
        "distribution so codecs cannot become a shortcut: random 8th-order Butterworth low-pass "
        "filtering at 4–7.8 kHz (p=0.3), codec round trips via FFmpeg from {µ-law 8k, A-law 8k, "
        "GSM, Opus 8/16/24k, MP3 32k} (p=0.4, with <b>G.726 32k held out completely as an unseen "
        "test codec</b>), random gain ±6 dB, and light Gaussian noise (p=0.2). Sampling balances "
        "50% genuine (equal per corpus) and 50% fake (26% target+related, 12% diverse, 7% Hindi, "
        "5% VC).",
    )
    add_p(
        story,
        "For reproducibility, staged preparation writes <font name='Mono'>PREPARED.json</font> "
        "only after verifying every shard's row count and SHA-256 digest (fingerprint "
        "<font name='Mono'>f8eb0fe0…</font>). Before testing, <font name='Mono'>signoff.json</font> "
        "hashes the protocol, code commit (<font name='Mono'>a4a7068</font>), dataset fingerprint, "
        "selected model weights, frozen cutoff, and every test row + audio shard byte. Evaluation "
        "refuses to run if any hash drifts and preserves a completed result on retry. Uncertainty "
        "is quantified using both <b>Wilson 95% confidence intervals</b> and <b>2,000-resample "
        "speaker/voice cluster bootstrap 95% intervals</b>. Release v1 is scored on the exact "
        "same test rows as our scientific control.",
    )

    add_h3(story, "3.3.4  Ethical and Privacy Considerations")
    add_p(
        story,
        "All datasets used are public research corpora under their respective academic licenses "
        "(CC BY 4.0, CC BY-NC, CC BY-NC-SA); no private personal recordings were scraped or "
        "synthesized without consent. In the web app, uploaded files are read in memory, never "
        "saved to a database, never written to logs, and deleted in a <font name='Mono'>finally</font> "
        "block right after scoring. Every result card displays a permanent caveat reminding users "
        "that an automated score is an estimate—not legal proof—and advising them to verify "
        "suspicious calls through another channel.",
    )

    add_h2(story, "3.4  Proposed System Overview")
    add_p(
        story,
        "VoiceSleuth separates <b>model research</b> from <b>application delivery</b> cleanly. "
        "The research pipeline (<font name='Mono'>detector/</font> and <font name='Mono'>pipeline_v3/</font>) "
        "handles data auditing, training, calibration, and sealed evaluation. The web app "
        "(<font name='Mono'>app/</font>) imports any compatible model through a single Python "
        "interface (<font name='Mono'>Detector.analyze(data, filename)</font>) where the model "
        "owns audio decoding so serving matches training exactly.",
    )

    add_h2(story, "3.5  Compute Constraints, Resource Management, and Limitations")
    add_p(
        story,
        "We encountered practical compute constraints during the semester: a workstation power "
        "outage caused Colab to prune our first A100 session, requiring a replacement A100 and a "
        "pre-planned 4,500-step schedule for C3; and serial CPU <font name='Mono'>ffmpeg</font> "
        "calls inside multi-channel evaluation stretched the 5-channel test run to 108 minutes. "
        "Beyond compute, we record six scientific limitations honestly:",
    )
    add_bullets(story, [
        "<b>Familiar-family evaluation:</b> All four target families appeared in training on "
        "disjoint en_US texts and half-A voices; zero-shot detection on brand-new unseen TTS "
        "architectures is not claimed.",
        "<b>Unmatched test genuine/fake parents:</b> M-AILABS en_UK genuine recordings (Elizabeth "
        "Klett) were not available on Drive, so genuine FA is measured on independent corpora "
        "(LibriTTS, VCTK, AMI, EdAcc, IndieFake).",
        "<b>Low voice diversity on Chatterbox & FishTTS test cells (2 clusters each):</b> Their "
        "bootstrap intervals reflect unseen texts/parents rather than diverse speakers, whereas "
        "ElevenLabs-v3 (11 unseen voices) and Gemini (13 unseen voices) test voice generalization.",
        "<b>Advisory shortcut cues:</b> Pre-training train→dev simple-feature probes flagged "
        "acoustic separation above 0.80 AUC on joint features for all four targets (0.81–0.89) "
        "and on spectral tilt and zero-crossing rate for ElevenLabs-v3 and Gemini.",
        "<b>4-second minimum & simulated codecs only:</b> Clips shorter than 4 seconds after "
        "silence trimming are excluded in V2-90, and real cellular/WhatsApp calls were not yet "
        "recorded and tested.",
        "<b>Unresolved cross-corpus IDs:</b> Speaker overlap between LibriTTS/VCTK and Release "
        "v1's ASVspoof 5 (MLS) training speakers cannot be checked because ASVspoof 5 uses "
        "anonymized IDs.",
    ])

    # ==================================================================
    # CHAPTER 4: SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
    # ==================================================================
    add_chapter(story, "4  SOFTWARE REQUIREMENTS SPECIFICATION (SRS)")

    add_h2(story, "4.1  Introduction")
    add_p(
        story,
        "This chapter specifies the functional, non-functional, hardware, software, and interface "
        "requirements for the VoiceSleuth web application, adapted from our Product Requirements "
        "Document (<font name='Mono'>docs/PRD_deepfake_audio_app.md</font>).",
    )

    add_h2(story, "4.2  Overall Description")
    add_p(
        story,
        "VoiceSleuth serves three primary user groups: (1) everyday users checking suspicious "
        "WhatsApp voice notes or forwarded clips, (2) journalists or fact-checkers seeking a fast "
        "second opinion on viral audio, and (3) students and researchers inspecting segment-level "
        "model behavior. Because detection models are probabilistic, the UI never outputs a blunt "
        "'REAL / FAKE' stamp. Instead, it uses four explicit, plain-language verdict states "
        "(Table 4.1) plus an experimental banner whenever a research prototype backend is active.",
    )
    add_table(
        story,
        [
            ["Verdict Code", "User-Facing Title", "User-Facing Explanation"],
            ["likely_fake", "Likely AI-generated", "Large parts of this recording show strong signs of synthetic speech."],
            ["uncertain", "Can't tell", "The signs are mixed. Treat this recording with caution and verify it another way."],
            ["likely_real", "No strong signs of AI", "We didn't find strong signs of synthetic speech. This doesn't prove the recording is genuine."],
            ["insufficient_audio", "Not enough speech", "We couldn't find enough clear speech to analyze. Try a longer or clearer recording."],
        ],
        [3.1, 3.6, 9.2],
        "Table 4.1: Exact four-state verdict copy enforced in VoiceSleuth's frontend.",
    )

    add_h2(story, "4.3  Functional Requirements")
    add_bullets(story, [
        "<b>FR1 (File Upload & Validation):</b> Users can drag-and-drop or browse audio/video "
        "files (.wav, .mp3, .m4a, .aac, .ogg, .opus, .flac, .webm, .mp4, .mov). Files over 50 MB "
        "are rejected in the browser without a network request and again on the server. Files "
        "longer than 10 minutes or unreadable/corrupt files are rejected before model inference.",
        "<b>FR2 (Background Analysis & Cancellation):</b> Uploads return HTTP 202 with a job ID. "
        "The UI shows Uploading → Analyzing → Done states by polling every 1 second, and lets the "
        "user cancel an active or queued job.",
        "<b>FR3 (Verdict, Waveform & Segment Timeline):</b> Results show the verdict card, fake "
        "probability percentage, duration/speech stats, a Web Audio SVG waveform, and a "
        "color-coded segment bar (green &lt;33%, amber 33–66%, red ≥67%). Clicking any segment "
        "seeks local playback of the user's file directly in the browser.",
        "<b>FR4 (Segment Table & JSON Report):</b> Users can view a table of per-segment start "
        "time, end time, and probability, and download a clean JSON report containing metadata "
        "and scores (with zero audio bytes inside).",
        "<b>FR5 (Pluggable Detector Contract):</b> Changing <font name='Mono'>DETECTOR_BACKEND</font> "
        "switches between <font name='Mono'>mock</font>, <font name='Mono'>legacy</font>, "
        "<font name='Mono'>detector</font> (Release v1), and <font name='Mono'>v3</font> without "
        "editing any web app code.",
    ])

    add_h2(story, "4.4  Non-Functional Requirements")
    add_bullets(story, [
        "<b>Performance:</b> Upload endpoint responds within 1 s after body transfer; MockDetector "
        "finishes a 1-minute clip in &lt;2 s; GPU inference on a 1-minute clip targets &lt;15 s.",
        "<b>Reliability:</b> A single corrupt file or worker exception marks only that job as "
        "failed and never crashes the server. If model loading fails at startup, health returns "
        "<font name='Mono'>model_loaded: false</font> and analyze returns HTTP 503.",
        "<b>Security:</b> Files are validated by decoding/probing, not by extension alone. FFmpeg "
        "and ffprobe run via argument lists (never a shell) with strict protocol and format "
        "whitelists. Responses set <font name='Mono'>X-Content-Type-Options: nosniff</font>, "
        "<font name='Mono'>Referrer-Policy: no-referrer</font>, strict Content-Security-Policy, "
        "and <font name='Mono'>Cache-Control: no-store</font> on all API routes.",
        "<b>Privacy:</b> Zero database storage of audio, zero raw audio in server logs, immediate "
        "memory clearing when a job starts or cancels, and 60-minute TTL purging of finished JSON "
        "results.",
        "<b>Accessibility & Usability:</b> Responsive layout down to 360 px width, automatic "
        "light/dark mode following OS preference, keyboard navigation, and text+icon verdicts "
        "(never color alone).",
    ])

    add_h2(story, "4.5  Hardware and Software Requirements")
    add_h3(story, "4.5.1  Developer and Training Environment")
    add_table(
        story,
        [
            ["Component", "Specification Used"],
            ["Local Workstation", "Linux (x86_64), AMD Ryzen 5 / NVIDIA RTX 5060 GPU, Python 3.14 venv, Node.js + npm"],
            ["Cloud Training GPUs", "Google Colab NVIDIA L4 (24 GB VRAM, 12 vCPU) and A100-SXM4 (40 GB VRAM, 12 vCPU)"],
            ["Core ML Stack", "PyTorch (CUDA + bf16 autocast), Hugging Face Transformers, SciPy, scikit-learn, soundfile, pyarrow"],
            ["Web App Stack", "FastAPI, Uvicorn, python-multipart, Pydantic v2, React 18, TypeScript 5.9, Vite 7, Vitest"],
            ["Media Tooling", "System FFmpeg and ffprobe (PCM, MP3, AAC, Opus, G.711 µ-law, G.726, GSM support)"],
        ],
        [4.2, 11.7],
        "Table 4.2: Hardware and software environment used for training, evaluation, and app development.",
    )

    add_h3(story, "4.5.2  Client Environment (End User)")
    add_p(
        story,
        "Any current desktop or mobile browser (Chrome, Firefox, or Safari) with JavaScript and "
        "Web Audio API support, screen width ≥360 px, and standard HTTP/HTTPS connectivity to the "
        "host server. No browser extensions, plugins, or accounts are required.",
    )

    add_h2(story, "4.6  External Interface Requirements")
    add_p(
        story,
        "The web app communicates strictly over JSON REST endpoints under <font name='Mono'>/api/</font>, "
        "while the backend connects to detection models through the typed Python protocol defined "
        "in <font name='Mono'>app/backend/detectors/base.py</font>. Every detector exposes "
        "<font name='Mono'>info: ModelInfo</font> and <font name='Mono'>analyze(data: bytes, filename: str) "
        "-> AnalysisResult</font>, raising <font name='Mono'>ValueError</font> on undecodable input.",
    )

    # ==================================================================
    # CHAPTER 5: SYSTEM DESIGN AND ARCHITECTURE
    # ==================================================================
    add_chapter(story, "5  SYSTEM DESIGN AND ARCHITECTURE")

    add_h2(story, "5.1  Introduction")
    add_p(
        story,
        "VoiceSleuth is built around a clean separation of concerns: the browser handles user "
        "interaction, local audio decoding for waveform display, and local playback; the FastAPI "
        "server handles strict upload validation, job queueing, and security headers; and the "
        "selected detector module owns all audio decoding and neural inference.",
    )

    add_h2(story, "5.2  System Architecture Overview")
    add_p(
        story,
        "Figure 5.1 shows the full four-layer architecture. Running FastAPI as a single process "
        "with an in-memory <font name='Mono'>ThreadPoolExecutor</font> (default 1 worker on GPU "
        "to avoid VRAM contention, 2 workers on CPU) keeps deployment simple and guarantees the "
        "model is loaded into memory only once at startup. In production, FastAPI also serves the "
        "compiled React bundle from <font name='Mono'>app/frontend/dist</font> directly at "
        "<font name='Mono'>/</font>.",
    )
    story.append(Spacer(1, 3))
    story.append(draw_architecture_diagram())
    story.append(Paragraph("Figure 5.1: VoiceSleuth four-layer web and pluggable detector architecture.", STYLES["caption"]))

    add_h2(story, "5.3  Data Flow Design")
    add_p(
        story,
        "A key engineering rule from our model work is that <b>the app never resamples or "
        "re-encodes user audio</b> before handing it to the detector. Early experiments taught us "
        "that even tiny differences between training-time and serving-time decoding shift model "
        "scores. Figure 5.2 traces how raw uploaded bytes move through incremental memory "
        "parsing, ffprobe duration checking, canonical model decoding, and back to the browser as "
        "a lightweight JSON result while temporary files are unlinked immediately.",
    )
    story.append(Spacer(1, 3))
    story.append(draw_dfd_diagram())
    story.append(Paragraph("Figure 5.2: End-to-end data flow from raw upload to local browser playback.", STYLES["caption"]))

    add_h2(story, "5.4  Sequence Diagram")
    add_p(
        story,
        "Figure 5.3 illustrates the asynchronous request lifecycle. Because deep learning "
        "inference on a long recording can take several seconds, <font name='Mono'>POST /api/analyze</font> "
        "returns HTTP 202 immediately after validating the file and enqueuing a job. The browser "
        "polls <font name='Mono'>GET /api/jobs/{job_id}</font> every second until the status "
        "becomes <font name='Mono'>done</font>, <font name='Mono'>failed</font>, or "
        "<font name='Mono'>cancelled</font>.",
    )
    story.append(Spacer(1, 3))
    story.append(draw_sequence_diagram())
    story.append(Paragraph("Figure 5.3: Asynchronous job submission, inference, and polling sequence.", STYLES["caption"]))

    add_h2(story, "5.5  Frontend Component Design")
    add_p(
        story,
        "The React frontend (<font name='Mono'>app/frontend/src/</font>) avoids heavy charting "
        "or audio libraries and breaks the interface into focused, testable components:",
    )
    add_table(
        story,
        [
            ["Component", "Responsibility"],
            ["DropZone.tsx", "Drag-and-drop and file picker; enforces extension and 50 MB size limit client-side before upload."],
            ["ProgressView.tsx", "Displays Uploading and Analyzing status with an accessible Cancel button."],
            ["VerdictCard.tsx", "Renders the exact four-state title, icon, fake probability percentage, and explanation."],
            ["Recording.tsx", "Decodes the user's local File via Web Audio API into an SVG waveform and binds local audio playback."],
            ["Timeline.tsx", "Renders clickable green/amber/red probability bars aligned to duration with ARIA labels."],
            ["SegmentTable.tsx", "Lists exact start time, end time, and fake probability per segment; supports JSON download."],
            ["Caveat.tsx & ModelBadge.tsx", "Always-visible accuracy caveat, active model name/version, and experimental warning banner."],
        ],
        [4.2, 11.7],
        "Table 5.1: Frontend React component layout in app/frontend/src/components/.",
    )

    add_h2(story, "5.6  Backend API Design")
    add_table(
        story,
        [
            ["Endpoint", "Method", "Response & Error Codes"],
            ["/api/health", "GET", "200 {status: 'ok', model_loaded: bool}."],
            ["/api/model", "GET", "200 {model: ModelInfo, limits: {max_upload_mb, max_duration_s, accepted_extensions}}; 503 if unavailable."],
            ["/api/analyze", "POST", "202 {job_id, status: 'queued'}. Errors: 400 unsupported_type, 413 too_large, 422 too_long/unreadable, 503 busy/model_unavailable."],
            ["/api/jobs/{job_id}", "GET", "200 job status, progress, result or error, timestamps; 404 not_found if unknown or expired."],
            ["/api/jobs/{job_id}", "DELETE", "204 No Content — cancels queued/running job or removes finished result immediately."],
        ],
        [3.6, 1.7, 10.6],
        "Table 5.2: VoiceSleuth REST API contract implemented in app/backend/main.py.",
    )

    # ==================================================================
    # CHAPTER 6: IMPLEMENTATION DETAILS
    # ==================================================================
    add_chapter(story, "6  IMPLEMENTATION DETAILS")

    add_h2(story, "6.1  Introduction")
    add_p(
        story,
        "Rather than jumping straight to one final script, we built VoiceSleuth in three clear "
        "iterations between August and November 2026. Each stage exposed real flaws in data "
        "quality, model transfer, or code, and directly shaped the next version.",
    )

    add_h2(story, "6.2  Technology Stack Implementation")
    add_p(
        story,
        "The repository uses a pinned Python 3.14 virtual environment (<font name='Mono'>venv/</font>) "
        "for backend services and local testing, paired with Google Colab Linux GPU runtimes for "
        "large-scale SSL training. All model pipelines are strictly isolated by folder ownership: "
        "<font name='Mono'>deepfake_model/</font> holds the frozen Gen 1 CNN-LSTM work, "
        "<font name='Mono'>detector/</font> holds the frozen Gen 2 Release v1 pipeline and its "
        "inference adapter, <font name='Mono'>pipeline_v3/</font> holds the Gen 3 V2-90 and "
        "bilingual training code, and <font name='Mono'>app/</font> holds all FastAPI and React "
        "application code.",
    )

    add_h2(story, "6.3  Detection Pipeline Evolution (Gen 1 → Gen 2 → Gen 3)")

    add_h3(story, "6.3.1  Generation 1: Auditing the Legacy CNN-LSTM Baseline")
    add_p(
        story,
        "Our starting point was a 3-block CNN + Bidirectional LSTM trained on 128×94 log-Mel "
        "spectrograms of 1-second windows. When we audited the historical run, we discovered three "
        "serious bugs: the old fold loader mixed rows across speakers, 117 duplicate audio groups "
        "crossed speaker folders, and 18 recordings had near-silent openings that collapsed into "
        "identical -80 dB floor features. We rebuilt the speaker-disjoint split (29 train / 10 dev "
        "/ 10 test folders), enforced identical 16 kHz mono decoding across training and "
        "inference, and froze the decision rule (cumulative mean ≥ 0.5996) before scoring test "
        "speakers once. While in-corpus IndieFake test AUC reached 0.9939, external performance "
        "collapsed (23.4% EER on In-the-Wild and only 6/22 fakes caught on VA), proving that "
        "handcrafted spectrograms from one corpus do not generalize.",
    )

    add_h3(story, "6.3.2  Generation 2: Frozen XLS-R 300M (Release v1)")
    add_p(
        story,
        "Next, we built <font name='Mono'>detector/</font> around a frozen pretrained SSL "
        "backbone. Comparing WavLM-Base+ (dev EER 8.1%) against Meta's "
        "<font name='Mono'>wav2vec2-xls-r-300m</font> (dev EER 5.1%) on ASVspoof 5 [59] + "
        "IndieFake, XLS-R won cleanly. We trained three random seeds (42, 43, 44) of a learned "
        "softmax layer-mix plus attentive statistics pooling head using real codec, noise, gain, "
        "and edge-silence augmentation, then calibrated probabilities via Platt scaling "
        "(<font name='Mono'>p = sigmoid(0.688 * logit + 1.760)</font>) with thresholds set on dev "
        "(likely_real &lt; 0.514, likely_fake ≥ 0.56235). Release v1 shipped in "
        "<font name='Mono'>detector/inference.py</font> and achieved 11.3% EER on sealed "
        "In-the-Wild—but later tests showed it still missed modern commercial TTS like Gemini "
        "and ElevenLabs-v3.",
    )

    add_h3(story, "6.3.3  Generation 3: V2-90 Fine-Tuned Ensemble (C2, C3, and C4)")
    add_p(
        story,
        "To push modern TTS detection toward 90% per family, we implemented "
        "<font name='Mono'>pipeline_v3/v2_90.py</font> and trained two distinct fine-tuned "
        "candidates on the exact same verified training shards (<font name='Mono'>f8eb0fe0…</font>): "
        "<b>C2</b> fine-tunes all 24 transformer layers of Meta's "
        "<font name='Mono'>wav2vec2-xls-r-300m</font> (CNN feature extractor frozen, batch size "
        "32, encoder LR 2e-5, head LR 1e-3, OneCycleLR, 6,000 steps on L4); <b>C3</b> fine-tunes "
        "the top 24 of 48 transformer layers of NII Yamagishi Lab's "
        "<font name='Mono'>xls-r-1b-anti-deepfake</font> (rev. <font name='Mono'>60c96543</font>, "
        "lower 24 layers frozen, batch size 16, encoder LR 1e-5, 4,500 steps on A100). Both attach "
        "a learnable softmax mix over all hidden states followed by temporal mean+std pooling and "
        "a 2-layer GELU MLP head.",
    )
    add_p(
        story,
        "<b>Critical engineering bugs caught and fixed before full training:</b> During real-data "
        "preparation on L4, we hit a <font name='Mono'>KeyError: 'ami-ihm/FEO070/34'</font> "
        "caused by reused AMI parquet row IDs; commit <font name='Mono'>6c3c4c9</font> switched "
        "to unique <font name='Mono'>audio_id</font> keys and added a duplicate-UID guard. Even "
        "more importantly, our first GPU smoke test of C3 (<font name='Mono'>smoke3</font>) "
        "produced <font name='Mono'>NaN</font> development cutoffs because Hugging Face "
        "Transformers did not map the upstream Fairseq key prefixes "
        "(<font name='Mono'>m_ssl.model.*</font>), leaving encoder weights randomly initialized. "
        "In commit <font name='Mono'>a4a7068</font>, we wrote an explicit key mapper "
        "(<font name='Mono'>antideepfake_source_key()</font>) using <font name='Mono'>strict=True</font> "
        "loading across all 806 tensors and added finite-value assertions on losses and scores. "
        "The corrected smoke run (<font name='Mono'>smoke3fixed</font>) loaded all 806 tensors, "
        "resumed cleanly at step 20, and scored finite dev metrics through step 60.",
    )

    add_h2(story, "6.4  Calibration, Thresholding, and Cryptographic Sign-Off")
    add_p(
        story,
        "Because C2 and C3 output raw logits on different scales, our prespecified selection rule "
        "standardizes each candidate using its own development-genuine mean and standard deviation "
        "(C2: <font name='Mono'>mu = -4.6351, sd = 1.9971</font>; C3: <font name='Mono'>mu = "
        "-3.9402, sd = 1.2250</font>) and forms the equal-weight ensemble <b>C4</b>:",
    )
    add_p(story, "z_i(x) = (logit_i(x) - mu_i) / sd_i      |      score_C4(x) = 0.5 * (z_C2(x) + z_C3(x))", "formula")
    add_p(
        story,
        "On development clean audio, C2 alone (pooled 0.9777), C3 alone (0.9565), and C4 (0.9822) "
        "all achieved 100% minimum-target detection (<font name='Mono'>key = 1.0</font>); C4 won "
        "the prespecified pooled-detection tie-break. Next, <font name='Mono'>calibrate()</font> "
        "scored the 634 clean English calibration genuine clips (126 speakers) and picked the "
        "lowest threshold flagging at most <font name='Mono'>floor(0.15 * 634) = 95</font> clips: "
        "<b>0.8829656848957964</b> (flagging 95/634 = 14.98%). Finally, "
        "<font name='Mono'>signoff()</font> wrote the write-once cryptographic binding file "
        "<font name='Mono'>signoff.json</font> (SHA-256 <font name='Mono'>c879b05c…</font>) "
        "locking the protocol, code commit, dataset fingerprint, member weights, cutoff, and test "
        "shard bytes before a single test clip was scored.",
    )

    add_h2(story, "6.5  Privacy-First Upload Streaming and Frontend Playback")
    add_p(
        story,
        "Standard FastAPI <font name='Mono'>UploadFile</font> handlers automatically spool large "
        "uploads to temporary disk files, which could leave raw user audio behind if a request "
        "drops. Instead, <font name='Mono'>app/backend/uploads.py</font> streams the HTTP request "
        "directly into <font name='Mono'>python_multipart.MultipartParser</font> in RAM, caps "
        "file bytes at 50 MB and header overhead at 64 KB, and silences the parser's internal "
        "logger so malformed input bytes can never leak into server logs. On the frontend, "
        "<font name='Mono'>Recording.tsx</font> creates a local blob URL with "
        "<font name='Mono'>URL.createObjectURL(file)</font>, draws the waveform in the browser "
        "using the Web Audio API, and revokes the URL on unmount—so the server never sends audio "
        "back over the network.",
    )

    # ==================================================================
    # CHAPTER 7: SOFTWARE TESTING AND VALIDATION
    # ==================================================================
    add_chapter(story, "7  SOFTWARE TESTING AND VALIDATION")

    add_h2(story, "7.1  Introduction")
    add_p(
        story,
        "To make sure neither software bugs nor data-handling errors could quietly corrupt "
        "results, we tested VoiceSleuth at three levels: backend API integration tests, machine "
        "learning pipeline integrity tests, and frontend component tests, plus a real-data GPU "
        "checkpoint-resume smoke test before launching full training.",
    )

    add_h2(story, "7.2  Testing Strategies")
    add_bullets(story, [
        "<b>Backend API Testing (pytest + FastAPI TestClient):</b> Generates real test audio "
        "fixtures on the fly using system FFmpeg (3 s tone+noise in WAV, MP3, M4A, OGG/Opus, and "
        "MP4; a 0.3 s clip; a 12-minute clip; and a plain text file renamed .wav) and checks HTTP "
        "status codes, job lifecycle, queue limits, temp-file cleanup, and log privacy.",
        "<b>ML Pipeline & Integrity Testing (pytest):</b> Exercises shard verification, tamper "
        "detection, split-crossing rejection, AMI reused-ID handling, Fairseq-to-Transformers "
        "weight mapping, non-finite loss rejection, sign-off binding checks, and atomic "
        "evaluation preservation.",
        "<b>Frontend Unit Testing (Vitest + React Testing Library):</b> Verifies client-side "
        "50 MB rejection, exact four-state verdict copy rendering, and accessible timeline ARIA "
        "labels and color bands.",
        "<b>Real-Data GPU Smoke Test:</b> Ran <font name='Mono'>smoke3fixed</font> on Colab A100 "
        "with <font name='Mono'>--stop-after 20</font>, resumed from step 20 to step 60, and "
        "verified finite dev scores across all four targets with zero out-of-memory errors.",
    ])

    add_h2(story, "7.3  Test Cases and Verification Results")
    add_p(
        story,
        "Running our local test suites under Python 3.14 and Node/Vitest passes cleanly: "
        "<b>118/118 Python tests</b> (<font name='Mono'>app/backend</font> + "
        "<font name='Mono'>pipeline_v3/tests</font> on main branch), <b>110/110 V2-90 branch "
        "tests</b> (including 10 dedicated V2-90 integrity tests in "
        "<font name='Mono'>test_v2_90.py</font>), and <b>6/6 frontend tests</b> across 3 test "
        "files. Table 7.1 summarizes key verified test cases.",
    )
    add_table(
        story,
        [
            ["TC ID", "Module / Target", "Scenario Tested", "Expected Outcome", "Status"],
            ["TC-01", "app/backend API", "Upload valid WAV, MP3, M4A, OGG, MP4 fixtures", "HTTP 202 queued → done with valid JSON result", "PASS"],
            ["TC-02", "app/backend API", "Upload file >50 MB or >10 min duration", "Rejected with 413 too_large / 422 too_long", "PASS"],
            ["TC-03", "app/backend API", "Upload text file renamed .wav", "Rejected by ffprobe with 422 unreadable", "PASS"],
            ["TC-04", "app/backend Jobs", "Fill queue past MAX_QUEUE; cancel queued job; TTL expiry", "503 busy; cancelled frees bytes; 404 after TTL", "PASS"],
            ["TC-05", "app/backend Privacy", "Inspect temp dir & captured logs after success & failure", "Temp dir empty; zero raw audio bytes in logs", "PASS"],
            ["TC-06", "app/frontend UI", "DropZone >50 MB, VerdictCard 4 states, Timeline bands", "Client error; exact copy; correct ARIA labels", "PASS"],
            ["TC-07", "v2_90.py Prep", "Inject duplicate UID or speaker crossing train/test", "prep() raises ValueError and refuses publish", "PASS"],
            ["TC-08", "v2_90.py Signoff", "Alter shard byte or index after signoff.json written", "check_signoff() raises tamper ValueError", "PASS"],
            ["TC-09", "v2_90.py Model", "Load AntiDeepfake 806 tensors; reject NaN loss/scores", "Strict load passes; non-finite raises ValueError", "PASS"],
            ["TC-10", "Colab A100 Smoke", "Train C3 to step 20, stop, resume to step 60 on real data", "Resumes at 20, finite dev scores, 0 OOM", "PASS"],
        ],
        [1.4, 2.7, 4.9, 5.2, 1.7],
        "Table 7.1: Key automated software and ML pipeline test cases verified in VoiceSleuth.",
    )

    # ==================================================================
    # CHAPTER 8: RESULTS AND DISCUSSIONS
    # ==================================================================
    add_chapter(story, "8  RESULTS AND DISCUSSIONS")

    add_h2(story, "8.1  Introduction")
    add_p(
        story,
        "In this chapter we report all measured outcomes honestly and keep three things strictly "
        "separate: (1) development metrics used to pick models, (2) single-pass held-out results "
        "under frozen cutoffs, and (3) post-hoc descriptive analysis used to understand error "
        "behavior without altering the official verdict.",
    )

    add_h2(story, "8.2  Evolution Across Model Generations")
    add_p(
        story,
        "Before V2-90, we evaluated Generations 1 and 2 on two sealed external benchmarks—the "
        "31,712-clip <b>In-the-Wild</b> corpus [57] and the 44-clip <b>VA</b> collection. "
        "Moving from our audited Gen 1 CNN-LSTM to Gen 2 Release v1 (frozen XLS-R 300M + 3 heads "
        "+ codec augmentation) cut sealed In-the-Wild EER in half, from <b>23.4%</b> (AUC 0.839) "
        "down to <b>11.3%</b> (AUC 0.957, 95% speaker bootstrap [8.9%, 13.5%]). Speech length "
        "mattered greatly: on In-the-Wild clips with 4–16 seconds of speech (n=9,111), Release v1 "
        "reached <b>4.4% EER</b> and caught <b>98.3%</b> of fakes (1.5% EER above 16 s), whereas "
        "clips under 4 seconds dropped to 14.8% EER and 59.5% fake recall. On sealed VA, Release "
        "v1 caught 19/22 fakes at 1/22 false alarm (6.8% EER vs. 6/22 fakes for Gen 1).",
    )
    add_p(
        story,
        "Despite that strong In-the-Wild result, testing Release v1 on modern 2025–2026 TTS "
        "revealed a blind spot: it missed most Gemini-3.1-Flash-TTS and ElevenLabs-v3 audio, "
        "which directly motivated our V2-90 fine-tuning experiment.",
    )

    add_h2(story, "8.3  V2-90 Held-Out Results and Release v1 Comparison")
    add_p(
        story,
        "Both V2-90 candidates finished their planned schedules cleanly (C3: 4,500 steps on A100 "
        "at 01:11 IST; C2: 6,000 steps on L4 at 01:16 IST). One subtle code behavior should be "
        "noted transparently: both C2 and C3 hit 100% minimum-target detection on dev "
        "(<font name='Mono'>key = 1.0</font>) at step 500, and because "
        "<font name='Mono'>best.pt</font> updates only on strict <font name='Mono'>&gt;</font>, "
        "both froze their step-500 weights even though dev pooled detection climbed to 0.9989 (C2) "
        "and 0.9844 (C3) later in training. Under our frozen protocol, we evaluated the exact "
        "selected C4 ensemble (<font name='Mono'>c2 + c3</font> step-500 z-mean) at the frozen "
        "cutoff <b>0.882966</b> without changing anything.",
    )
    add_p(
        story,
        "<b>Primary Clean English Verdict — MISS (on false-alarm budget, not fake detection):</b> "
        "Table 8.1 gives the complete held-out clean results with Wilson and 2,000-resample "
        "cluster bootstrap 95% confidence intervals. All four target modern TTS families easily "
        "beat the 90% goal (<b>98.33%–100.00%</b>, pooled <b>758/760 = 99.74%</b>). However, "
        "English genuine false alarms reached <b>19.15% (158/825)</b>, exceeding the 15.0% cap "
        "(123/825) by <b>35 recordings</b>.",
    )

    add_table(
        story,
        [
            ["Held-Out Test Cell (Clean)", "Flagged / N", "Rate", "Wilson 95% CI", "Clusters", "Cluster Boot 95% CI"],
            ["<b>Chatterbox</b> (target ≥90%)", "232 / 232", "<b>100.00%</b>", "[98.37%, 100.0%]", "2", "[100.0%, 100.0%]"],
            ["<b>ElevenLabs-v3</b> (target ≥90%)", "118 / 120", "<b>98.33%</b>", "[94.13%, 99.54%]", "11", "[95.59%, 100.0%]"],
            ["<b>FishTTS</b> (target ≥90%)", "283 / 283", "<b>100.00%</b>", "[98.66%, 100.0%]", "2", "[100.0%, 100.0%]"],
            ["<b>Gemini-3.1-Flash-TTS</b> (target ≥90%)", "125 / 125", "<b>100.00%</b>", "[97.02%, 100.0%]", "13", "[100.0%, 100.0%]"],
            ["<b>English Genuine (pooled, ≤15%)</b>", "<b>158 / 825</b>", "<b>19.15%</b>", "[16.61%, 21.98%]", "150", "[14.14%, 24.43%]"],
            ["  • genuine/libritts", "112 / 391", "28.64%", "[24.39%, 33.32%]", "71", "[19.60%, 38.21%]"],
            ["  • genuine/edacc", "31 / 177", "17.51%", "[12.62%, 23.79%]", "36", "[12.25%, 23.16%]"],
            ["  • genuine/indiefake", "14 / 98", "14.29%", "[8.70%, 22.56%]", "11", "[1.94%, 32.22%]"],
            ["  • genuine/ami", "1 / 24", "4.17%", "[0.74%, 20.24%]", "12", "[0.00%, 15.00%]"],
            ["  • genuine/vctk", "0 / 135", "0.00%", "[0.00%, 2.77%]", "20", "[0.00%, 0.00%]"],
            ["IndieFake fakes (descriptive)", "95 / 103", "92.23%", "[85.42%, 96.01%]", "12", "[81.65%, 100.0%]"],
            ["<b>Hindi Genuine (secondary)</b>", "<b>97 / 1,298</b>", "<b>7.47%</b>", "[6.16%, 9.03%]", "267", "[5.09%, 10.17%]"],
            ["IndicSynth/freevc24 (Hindi VC)", "149 / 150", "99.33%", "[96.32%, 99.88%]", "15", "[98.00%, 100.0%]"],
            ["hi__IndicF5 (Hindi TTS)", "36 / 37", "97.30%", "[86.18%, 99.52%]", "36", "[91.67%, 100.0%]"],
            ["hi__OmniVoice/Veena/VoxCPM2/Voxtral", "125 / 125", "100.00%", "[87.1%–90.4%, 100%]", "2–29", "[100.0%, 100.0%]"],
        ],
        [4.9, 1.9, 1.6, 3.0, 1.4, 3.1],
        "Table 8.1: V2-90 C4 held-out clean results at frozen cutoff 0.882966 (sign-off c879b05c).",
    )

    add_p(
        story,
        "<b>Why did English false alarms cross 15%?</b> Breaking down the 158 false alarms reveals "
        "a sharp domain effect: <b>LibriTTS audiobooks alone accounted for 112 of the 158 errors "
        "(28.64% of 391 clips)</b>, whereas VCTK had 0.00% FA (0/135) and AMI meetings had 4.17% "
        "(1/24). At the speaker level, 92 of 150 test English speakers had zero false alarms, "
        "while 13 of the 15 worst speakers (80%–100% FA) were LibriTTS readers. Because MLAAD "
        "fakes are synthesized from read audiobook prompts, certain close-mic audiobook recordings "
        "in LibriTTS drifted close to the fake side of the boundary (22.0% FA in calibration, "
        "rising to 28.6% in test).",
    )

    add_p(
        story,
        "<b>Apples-to-apples comparison with Release v1:</b> Scoring Release v1 on the exact same "
        "3,263 test clips highlights how large the modern-TTS gain is (Figure 8.1, Figure 8.2, and "
        "Table 8.2). At its shipped cutoff (0.56235), Release v1 caught only <b>3.2%</b> of "
        "Gemini-3.1-Flash-TTS and <b>9.2%</b> of ElevenLabs-v3 (45.9% pooled across the four "
        "targets). Even after independently calibrating Release v1 to ≤15% on the identical 634 "
        "English calibration clips (cutoff 0.02785), Release v1 still caught only <b>26.4%</b> of "
        "Gemini and <b>49.2%</b> of ElevenLabs-v3 (72.9% pooled)—while suffering <b>20.36%</b> "
        "English FA (worse than V2-90's 19.15%) and <b>36.59%</b> Hindi genuine FA (nearly five "
        "times worse than V2-90's 7.47%).",
    )

    add_figure(
        story,
        ASSETS / "fig1_model_comparison.png",
        "Figure 8.1: Clean-audio fake detection comparing V2-90 C4 against Release v1 (shipped and independently calibrated) on identical held-out clips.",
    )
    add_figure(
        story,
        ASSETS / "fig2_genuine_fa_by_source.png",
        "Figure 8.2: Genuine false-alarm rates by source corpus on clean audio under the matched ≤15% calibration rule.",
    )

    add_table(
        story,
        [
            ["Cell (Clean, Same 3,263 Clips)", "V2-90 C4 (cut 0.8830)", "Release v1 Shipped (0.56235)", "Release v1 Calibrated (0.02785)"],
            ["Chatterbox (n=232)", "<b>100.0%</b> (232/232)", "36.2% (84/232)", "78.0% (181/232)"],
            ["ElevenLabs-v3 (n=120)", "<b>98.3%</b> (118/120)", "9.2% (11/120)", "49.2% (59/120)"],
            ["FishTTS (n=283)", "<b>100.0%</b> (283/283)", "88.3% (250/283)", "99.3% (281/283)"],
            ["Gemini-3.1-Flash-TTS (n=125)", "<b>100.0%</b> (125/125)", "3.2% (4/125)", "26.4% (33/125)"],
            ["<b>Pooled 4 Targets (n=760)</b>", "<b>99.7% (758/760)</b>", "<b>45.9% (349/760)</b>", "<b>72.9% (554/760)</b>"],
            ["<b>English Genuine FA (n=825)</b>", "<b>19.15%</b> (158/825)", "4.48% (37/825)", "<b>20.36%</b> (168/825)"],
            ["<b>Hindi Genuine FA (n=1,298)</b>", "<b>7.47%</b> (97/1,298)", "5.55% (72/1,298)", "<b>36.59%</b> (475/1,298)"],
            ["Hindi MLAAD 5 systems (n=162)", "<b>99.4%</b> (161/162)", "45.7% (74/162)", "82.7% (134/162)"],
            ["IndicSynth/freevc24 (n=150)", "<b>99.3%</b> (149/150)", "70.7% (106/150)", "90.7% (136/150)"],
        ],
        [4.7, 3.7, 3.7, 3.8],
        "Table 8.2: Direct comparison of V2-90 C4 and Release v1 on the exact same 3,263 held-out test clips.",
    )

    add_h2(story, "8.4  Simulated Call Codecs, Hindi Speech, and Post-Hoc 2% FA Analysis")
    add_p(
        story,
        "All four simulated telephone and messaging channels were scored under the identical "
        "frozen cutoff 0.882966 (Table 8.3 and Figure 8.3). Even when 8 kHz codecs (G.711 µ-law, "
        "unseen G.726 32k, and GSM) wiped out every frequency above 4 kHz, target modern TTS "
        "detection stayed at <b>98.3%–100.0%</b>. This is encouraging evidence against a pure "
        "high-frequency shortcut. Under GSM, English genuine FA dropped to <b>15.15% (125/825)</b>—"
        "just 2 clips above the 15% cap—while Hindi genuine FA fell to <b>4.16% (54/1,298)</b>.",
    )

    add_table(
        story,
        [
            ["Channel", "Verdict (FA Gap)", "Chatterbox", "ElevenLabs", "FishTTS", "Gemini", "EN Gen FA", "HI Gen FA", "HI MLAAD"],
            ["clean", "MISS (+35)", "100.0%", "98.3%", "100.0%", "100.0%", "19.15%", "7.47%", "99.4%"],
            ["mulaw8k", "MISS (+51)", "100.0%", "100.0%", "100.0%", "100.0%", "21.09%", "7.55%", "100.0%"],
            ["g726 (unseen)", "MISS (+80)", "100.0%", "100.0%", "100.0%", "100.0%", "24.61%", "8.63%", "99.4%"],
            ["opus12", "MISS (+90)", "100.0%", "99.2%", "100.0%", "100.0%", "25.82%", "8.71%", "99.4%"],
            ["gsm", "MISS (+2)", "98.7%", "98.3%", "99.7%", "100.0%", "15.15%", "4.16%", "99.4%"],
        ],
        [2.3, 2.1, 1.7, 1.7, 1.5, 1.5, 1.7, 1.7, 1.7],
        "Table 8.3: V2-90 C4 held-out performance across clean audio and four simulated call codecs (cutoff 0.8830).",
    )
    add_figure(
        story,
        ASSETS / "fig3_codec_robustness.png",
        "Figure 8.3: Target detection vs. English and Hindi genuine false-alarm rates across all five channels.",
    )

    add_p(
        story,
        "<b>Post-hoc descriptive operating-point analysis (2%, 5%, 10% EN FA):</b> Because real "
        "consumer tools often need much lower false-alarm rates than 15%, we checked the saved "
        "<font name='Mono'>eval-scores.jsonl</font> post-hoc (strictly descriptive; does not alter "
        "the frozen MISS verdict). On clean audio, V2-90 C4 achieves a pooled 4-target "
        "<b>ROC-AUC of 0.9958</b> and <b>EER of 2.72%</b> (FishTTS 0.78%, Chatterbox 2.17%, "
        "Gemini 2.47%, ElevenLabs-v3 6.73%). As shown in Figure 8.4 and Table 8.4, at "
        "<b>1.94% English genuine FA (16/825)</b>—where VCTK, AMI, and EdAcc all have 0.0% FA "
        "and Hindi FA is 0.62%—clean pooled detection is still <b>95.92% (729/760)</b>, and at "
        "<b>4.97% English FA (41/825)</b>, every single target family exceeds 90% (91.7%–100.0%, "
        "pooled 98.42%).",
    )

    add_table(
        story,
        [
            ["Channel", "Pooled AUC / EER", "At ~2% EN FA (16/825)\nCB / EL / FT / GM (Pooled)", "At ~5% EN FA (41/825)\nCB / EL / FT / GM (Pooled)", "At ~10% EN FA\nPooled (HI FA)"],
            ["clean", "<b>0.9958 / 2.72%</b>", "96.6 / 85.8 / 99.7 / 96.0 (<b>95.9%</b>)", "99.1 / 91.7 / 100 / 100 (<b>98.4%</b>)", "99.6% (4.01%)"],
            ["mulaw8k", "0.9939 / 3.66%", "86.6 / 84.2 / 99.3 / 91.2 (91.7%)", "95.3 / 93.3 / 99.7 / 99.2 (97.2%)", "99.5% (2.85%)"],
            ["g726", "0.9929 / 4.48%", "84.9 / 82.5 / 98.6 / 93.6 (91.1%)", "92.7 / 92.5 / 99.7 / 98.4 (96.2%)", "99.0% (2.70%)"],
            ["opus12", "0.9930 / 3.97%", "87.1 / 81.7 / 98.9 / 89.6 (91.1%)", "97.0 / 89.2 / 99.7 / 97.6 (96.8%)", "98.8% (3.47%)"],
            ["gsm", "0.9877 / 5.62%", "75.0 / 79.2 / 97.5 / 87.2 (86.1%)", "87.5 / 90.8 / 98.2 / 95.2 (93.3%)", "98.2% (2.54%)"],
        ],
        [2.0, 2.4, 4.4, 4.4, 2.7],
        "Table 8.4: Post-hoc descriptive detection rates at 2%, 5%, and 10% English genuine FA (does not change frozen verdict).",
    )
    add_figure(
        story,
        ASSETS / "fig4_operating_points.png",
        "Figure 8.4: Post-hoc detection vs. English genuine false-alarm rate (2%–15%) across target families and GSM.",
    )

    add_p(
        story,
        "<b>Shortcut audit disclosure:</b> As required by Protocol §5, we report the pre-training "
        "advisory simple-feature probes (<font name='Mono'>audit.json</font>): the joint probe "
        "exceeded 0.80 AUC on all four targets (Chatterbox 0.822, FishTTS 0.813, Gemini 0.882, "
        "ElevenLabs-v3 0.890), with spectral tilt and zero-crossing rate also above 0.80 on "
        "ElevenLabs-v3 (0.822 / 0.873) and Gemini (0.829 / 0.818), while the pooled multi-source "
        "probe had zero red flags (0.638). Native duration and native sample rate had no red "
        "flags above 0.80. Maintaining 0.988–0.994 AUC under 8 kHz narrowband codecs confirms "
        "that C4 learned strong cues beyond simple high-frequency bandwidth differences, though "
        "testing on matched-parent genuine recordings remains an important next step.",
    )

    add_h2(story, "8.5  System Performance, Testing, and User Interface Evaluation")
    add_p(
        story,
        "On the application side, <font name='Mono'>DETECTOR_BACKEND=detector</font> successfully "
        "loaded Release v1 in FastAPI and reproduced sealed-test probabilities on uploaded files "
        "(e.g., 0.920 in the web app vs. 0.916 in batch evaluation on "
        "<font name='Mono'>Speaker-16_bonafide17.wav</font>; the 0.004 gap is bfloat16 batching). "
        "Oversized uploads (115 MB) were cleanly blocked with HTTP 413. All 118 backend/pipeline "
        "tests and 6 frontend tests pass in under 12 seconds locally. One practical engineering "
        "lesson from the V2-90 evaluation run is that running serial CPU "
        "<font name='Mono'>ffmpeg</font> twice per clip (once per ensemble member) caused the "
        "5-channel test to take 108 minutes on an A100; caching codec-converted waveforms once "
        "across 12 CPU workers will cut future evaluations to under 3 minutes.",
    )

    add_h2(story, "8.6  Achievement of Objectives and Hypothesis Evaluation")
    add_table(
        story,
        [
            ["Objective / Hyp.", "Measured Outcome", "Verdict"],
            ["Obj. 1 (Literature)", "54 distinct papers reviewed + 60 references; 6 research gaps mapped.", "Achieved"],
            ["Obj. 2 (Pipeline)", "Zero leakage across 8 checks; SHA-256 sign-off verified (bindings_ok=True).", "Achieved"],
            ["Obj. 3 / H1 & H2", "H1 supported (+26.8 to +53.8 pts vs. Release v1); H2 rejected at frozen cutoff (19.15% EN FA vs. ≤15%, LibriTTS shift).", "Partial (Formal MISS)"],
            ["Obj. 4 / H3 & H4", "H3 supported on detection (98.3%–100% across all 4 codecs incl. unseen G.726); H4 supported (99.4% HI fakes at 7.47% HI FA).", "Achieved"],
            ["Obj. 5 (Web App)", "FastAPI + React app built, tested (118 Py + 6 UI tests), zero audio stored.", "Achieved"],
        ],
        [3.1, 9.6, 3.2],
        "Table 8.5: Summary of Semester VII objectives and hypothesis outcomes.",
    )

    # ==================================================================
    # CHAPTER 9: CONCLUSION AND FUTURE SCOPE
    # ==================================================================
    add_chapter(story, "9  CONCLUSION AND FUTURE SCOPE")

    add_h2(story, "9.1  Conclusion")
    add_p(
        story,
        "In Semester VII, our team built and evaluated <b>VoiceSleuth</b> from first principles—"
        "starting with a 54-paper review, auditing an early CNN-LSTM baseline, shipping a "
        "calibrated XLS-R Release v1 detector, training and benchmarking our V2-90 fine-tuned "
        "ensemble (C2 + C3), and wrapping the entire workflow in a privacy-first FastAPI and React "
        "web application.",
    )
    add_p(
        story,
        "Our held-out evaluation delivered a clear, honest engineering verdict. V2-90 C4 solved "
        "Release v1's blind spot on modern TTS: detection jumped from 3.2% to 100.0% on "
        "Gemini-3.1-Flash-TTS and from 9.2% to 98.3% on ElevenLabs-v3, held 98.3%–100.0% across "
        "all four simulated telephone codecs (including unseen G.726), and achieved 99.4% Hindi "
        "TTS detection with only 7.47% Hindi false alarms. At the same time, keeping our "
        "pre-registered protocol intact exposed a real domain-calibration issue: LibriTTS "
        "audiobooks pushed English genuine false alarms to 19.15%, resulting in a formal "
        "<b>MISS</b> on the 15% cap even though post-hoc clean AUC reached 0.9958 (and 95.9% "
        "pooled detection at ~2% FA). Reporting that miss openly—instead of tuning the threshold "
        "after looking at test scores—gives us an exact, actionable roadmap for Semester VIII.",
    )

    add_h2(story, "9.2  Future Scope (Semester VIII Roadmap)")
    add_bullets(story, [
        "<b>1. Fix checkpoint selection and LibriTTS calibration:</b> Update "
        "<font name='Mono'>v2_90.py</font> to break ties on <font name='Mono'>(key, pooled, "
        "-EER)</font> so training does not freeze at step 500, and apply domain-balanced "
        "calibration (plus a wider 'Can't tell' abstention band) to bring English genuine FA "
        "safely below 2–5%.",
        "<b>2. Knowledge distillation for fast laptop CPU serving:</b> Distill the heavy C2+C3 "
        "teacher ensemble (~1.3B combined parameters) into both a compact acoustic student and a "
        "4–6-block quantized XLS-R student (targeting ≤90 MB and ≤3 s CPU latency for 30 s audio) "
        "using <font name='Mono'>torch.export</font> / ONNX Runtime to replace deprecated "
        "TorchScript.",
        "<b>3. Collect a fresh sealed real-world test panel:</b> Record consented genuine and "
        "cloned clips passed over actual WhatsApp voice notes and cellular calls in English and "
        "Hindi, plus at least two completely held-out unseen modern TTS families.",
        "<b>4. Parallelize evaluation codecs:</b> Cache FFmpeg-converted test waveforms once "
        "across CPU workers to drop 5-channel evaluation time from 108 minutes to ~3 minutes.",
        "<b>5. Complete P1 web app features:</b> Add browser microphone recording (up to 60 s), "
        "Docker packaging, in-memory IP rate limiting, an end-to-end Playwright test, and a UI "
        "hint advising users to upload at least 5–10 seconds of clear speech.",
    ])

    # ==================================================================
    # CHAPTER 10: REFERENCES (60 verified sources — exceeds DU >=50 rule)
    # ==================================================================
    add_chapter(story, "10  REFERENCES")
    add_p(
        story,
        "The following 60 references include all 54 peer-reviewed articles examined in our "
        "Semester VII literature review (2023–2027) together with the 6 primary speech encoders "
        "and benchmark corpora used in our experiments.",
    )
    story.append(Spacer(1, 0.2 * cm))
    for ref in REFERENCES:
        story.append(Paragraph(ref, STYLES["ref"]))

    doc = AcademicDoc(OUT_PDF, TITLE)
    doc.multiBuild(story)
    print(f"Built {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_report()
