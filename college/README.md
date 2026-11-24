# College folder

| File | What it is | For |
|---|---|---|
| [1_DEMO_REPORT.md](1_DEMO_REPORT.md) | The project in simple words: problem, how it works, honest results, limitations, a 5-minute demo script, likely questions | Demo and presentation |
| [2_DATASET_RESEARCH.md](2_DATASET_RESEARCH.md) | Claude's research on newer datasets (SpeechFake, MLAAD, IndicSynth, VoxENES 2026, Deepfake-Eval-2024), with sizes, licences, download commands and a short download prompt for Codex | Downloading data at college |
| [3_CODEX_RESEARCH_PROMPT.md](3_CODEX_RESEARCH_PROMPT.md) | A prompt for Codex to do full dataset research and redesign the pipeline with full authority | Planning version 2 |
| [4_HOW_THE_PIPELINE_WORKS.pdf](4_HOW_THE_PIPELINE_WORKS.pdf) | 18-page plain-language walkthrough of the code: data, audit, shortcuts, preprocessing, augmentation, model, training, calibration, release, sealed test, app, and the decisions you can make | Understanding the pipeline before choosing next steps |
| [5_LITERATURE_MATRIX.md](5_LITERATURE_MATRIX.md) | All 54 downloaded papers classified by theme and relevance, In-the-Wild comparison table, gaps, missing foundation papers | Literature review |
| [6_PROJECT_PLAN_FROM_LITERATURE.pdf](6_PROJECT_PLAN_FROM_LITERATURE.pdf) | 12-page plan in the mentor's order: scope, problem statement, literature review method, research design, data collection, timeline, checklist | What to do next and how |
| [RESEARCH_V2.md](RESEARCH_V2.md), [DOWNLOAD_PLAN.sh](DOWNLOAD_PLAN.sh), [DOWNLOAD_STATUS.py](DOWNLOAD_STATUS.py) | Codex's v2 research, the dataset download script and its progress monitor | Data collection |

`make_pipeline_pdf.py` and `make_project_plan_pdf.py` regenerate the PDFs (shared styles in `pdf_style.py`; needs `pip install reportlab`).

Related: `../detector/RESULTS.md` (full v1 numbers) and `../docs/SETUP_LAPTOP.md` (running the app on a laptop).
