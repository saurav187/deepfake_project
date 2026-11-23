# College folder

| File | What it is | For |
|---|---|---|
| [1_DEMO_REPORT.md](1_DEMO_REPORT.md) | The project in simple words: problem, how it works, honest results, limitations, a 5-minute demo script, likely questions | Demo and presentation |
| [2_DATASET_RESEARCH.md](2_DATASET_RESEARCH.md) | Claude's research on newer datasets (SpeechFake, MLAAD, IndicSynth, VoxENES 2026, Deepfake-Eval-2024), with sizes, licences, download commands and a short download prompt for Codex | Downloading data at college |
| [3_CODEX_RESEARCH_PROMPT.md](3_CODEX_RESEARCH_PROMPT.md) | A prompt for Codex to do full dataset research and redesign the pipeline with full authority | Planning version 2 |
| [4_HOW_THE_PIPELINE_WORKS.pdf](4_HOW_THE_PIPELINE_WORKS.pdf) | 18-page plain-language walkthrough of the code: data, audit, shortcuts, preprocessing, augmentation, model, training, calibration, release, sealed test, app, and the decisions you can make | Understanding the pipeline before choosing next steps |

`make_pipeline_pdf.py` regenerates the PDF (`venv/bin/python college/make_pipeline_pdf.py`, needs `pip install reportlab`).

Related: `../detector/RESULTS.md` (full v1 numbers) and `../docs/SETUP_LAPTOP.md` (running the app on a laptop).
