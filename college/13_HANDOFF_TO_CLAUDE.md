# Pipeline v3 handoff for Claude

**24 November 2026 · Local branch `pipeline-v3-scratch` · Base `detector-v2` at `f1f7e59`**

## Stage 0 handoff · 25 November 2026

Use the [Stage 0 report, exact commands and resource estimates](17_STAGE0_REPORT.md) for the current `v3-data` session. The owner reports Drive is mounted. Refresh the code bundle from this branch first. The earlier LibriSpeech commercial recipe is now an optional Stage 3 add-on.

The complete local listing replay joined 28,539 genuine PhonemeDF recordings and transcripts to all 85,617 synthetic renditions. The 251 PhonemeDF readers share no IDs with the 73 reserved LibriSpeech dev readers. None of the 28,537 distinct normalized transcripts crosses reader folds. Accepted English counts await the resumable target-voice inference job and four-second crop check. Chatterbox is held out for transfer. Unknown voices are excluded; compact Melo built-in clusters form a single training cohort.

**Decision before GPU spending.** The unchanged Hindi selection gives a source-only AUC of 1.00 on 811 genuine and 1,596 fake selection recordings. This exceeds the requested gate of 0.60; bilingual training therefore remains blocked. The English preparation can proceed independently. Resolve the Hindi data composition or explicitly revise the research protocol before spending on a bilingual pilot. Preserve the one-source-reader denominator for the 210 Hindi FreeVC24 selection fakes. Stage 0 ran locally on metadata and test fixtures. All 53 backend and pipeline tests passed (five warnings), and the fetch pins matched files.tsv. There are no cloud voice labels, passed audio audit or trained v3 results from this step.

## English repair after the first Drive run

Use the [English repair report and commands](15_ENGLISH_DATA_FIX.md) before packing. Local replay matched all 16,201 saved metadata rows. Only three US readers have parents on Drive. All 1,333 UK references add one other reader. A family-blocked fallback retained 1,016 English fakes but still left selection and validation empty. Keep MLAAD as a diagnostic panel.

The new pilot uses OpenSLR dev-clean and dev-other (about 0.65 GB total), with a projected 768/192/192/144/144 English pairs in train/select/calibrate/threshold/validate. The plan requires 60 distinct readers, disjoint texts and stock voices, and 1,104 Azure plus 336 Gemini clips. The owner runs acquisition and generation from the updated setup notebook. The planner checks real transcript and member eligibility before making a jobs CSV. No repaired English clips have yet been generated or scored in this work. Pilot transfer holds out Gemini and FreeVC24. Hindi selection contains four real readers overall, but its 210 FreeVC24 fakes have only one source reader. The release recipe admits Gemini training voices and reports its familiar-family status.

The combined backend and pipeline suite passed **47 tests** with five warnings after this repair. Local checks covered the metadata replay, quota planning, fold isolation, a missing-reader failure, paired-parent metrics, and archive-to-catalog conversion. Real LibriSpeech eligibility, provider availability, channel controls and detection accuracy still need the owner-run stages.

### Earlier progress update · 25 November 2026

The owner asked Codex to re-check the prompt and continue. The prompt has no working-tree edits, and its requirements match this handoff. A local adapter improvement now recognizes both Hindi Parquet audio column names (`audio` and `audio_filepath`); regression coverage was added. The v3 test suite passes **14 tests** and Ruff passes for the changed files. PyArrow is not installed in the local venv, so actual Parquet reads remain to be verified in the authorized Colab data setup. The earlier 41-test combined-suite result has not been rerun after this addition. No downloads, synthesis, cloud sessions, training or sealed-set access occurred.

The owner wants a small Hindi/English audio deepfake detector that catches modern TTS and voice cloning on short phone recordings. The revised prompt adds Google Colab CLI 0.8.0 alongside notebooks. The immediate decision is whether a cheap frozen-feature pilot on matched data justifies spending A100 time. There is a runnable implementation and a local diagnosis. Trained v3 weights, modern-family recall and a release decision still require the owner's Colab run. [Owner's prompt](7_CODEX_FULL_RESEARCH_AND_PIPELINE_PROMPT.md) · [Runbook](../pipeline_v3/README.md)

## What the measurements establish

- Recomputed single-head comparisons on **7,984 development recordings** from ASVspoof 5 and IndieFake. The median clean fake logit for 594 familiar IndieFake fakes stored at 24 kHz was **9.30**, above the **3.26** for 16 kHz fakes. Recovering high-rate examples and masking rate in x1 improved clean IndieFake EER from **7.44% to 4.09%** on these rows. A 24 kHz input by itself is a poor explanation for the owner's earlier Gemini misses.
- Ran the **released three-head v1**, on CPU, on **32 development recordings** and their 32 matched 16 kb/s Opus views. In the IndieFake panel, median fake probability on genuine recordings rose from **4.35% to 10.92%** after Opus. The familiar fakes moved from **99.79% to 99.44%**. This was a small paired codec test. Clean vs conversational style and previously unseen synthesis remain unresolved.
- The leading hypothesis is a mismatch in generator families and recording styles. The owner's earlier Gemini observations are historical. Establishing the cause requires fresh matched Gemini and Microsoft development recordings and a comparison of frozen probes, the original head and a handcrafted baseline. All reproduction commands and denominators are in the [diagnosis](8_DIAGNOSIS_AND_PROBLEM.md).
- An **untrained int8 student fixture** occupied **151,661 bytes**. On a desktop Ryzen 5 7600X using two threads, the bytes-to-verdict path took **0.045 s warm median** for synthetic 30 s PCM. This measures architecture cost; trained accuracy, installed runtime footprint, laptop performance and a mobile port remain open. [Design and benchmark conditions](12_PIPELINE_V3_DESIGN.md)

## What is available

| Deliverable | Where to look |
|---|---|
| 54 unique paper entries, full-text findings and comparison table | [Analysis](9_PAPER_ANALYSIS.md) and [CSV](9_paper_analysis.csv). 40 and 40b are distinct. The CSV records selected results and flags unverified fields. |
| Research gaps and evidence | [Gap table](10_RESEARCH_GAPS.md). Paper 34's 24.32% belongs to a cited RawNet2 baseline; VeriVoice reports 99.66% on its own OpenVoice corpus. Paper 15's frozen XLS-R plus a large head achieved 7.10% external ITW EER. |
| Data selection and acquisition | [Dataset and consent plan](11_DATASETS_V3.md). Minimum **35.409 decimal GB**: M-AILABS US 8.017, four English MLAAD families 1.359, Kathbath Hindi 15.976, IndicSynth Hindi 7.122, IndicVoices Hindi 2.935. These are transfer bytes. Effective matched counts must be measured from the authorised files. |
| Training and serving | [Runbook](../pipeline_v3/README.md). Modules cover source adapters, matching, shard packing/staging, shortcut tests, synthesis planning, pilot probes, teacher, distillation, int8 export, calibration, validation and CPU benchmark. |
| Colab and CLI entry points | [CPU setup](../colab/01_setup_and_data.ipynb), [pilot](../colab/02_pilot.ipynb), [teacher](../colab/03_teacher.ipynb), [student/export](../colab/04_student_export.ipynb), [evaluation](../colab/05_evaluate.ipynb), and [shared runner](../colab/run_stage.py). |
| App integration | The opt-in backend `DETECTOR_BACKEND=v3` reads `V3_RELEASE`. It handles uploaded bytes in anonymous Linux memory. The existing `DETECTOR_BACKEND=detector` selects v1. The exporter needs only the student and release metadata at serving time. |

## What to do next

1. **Use a CPU session for acquisition and joins.** Follow the three opening steps in the runbook. Inspect the dry-run transfer budget and obtain the owner's approval before acquisition. After matching, count real/fake clips, distinct source and target speakers, sentence families and generators in each language and fold. Missing reference voice metadata or too few US audiobook readers can empty a fold. Resolve this with additional matched, consented speech.
2. **Run the pilot before the teacher.** Start with 1,000 clips per class. The current decision rule requires held-family EER ≤20%, an advantage of ≥3 percentage points over the cepstral baseline, an Opus penalty ≤5 points, Hindi and English EER each ≤25%, and scoring coverage ≥95%. A passing nuisance audit is also required. Check absolute family counts and speaker intervals before interpreting a pass. [Pilot implementation](../pipeline_v3/pilot.py)
3. **Use new development clips for diagnosis.** Score clean and phone-delivered speech, human and synthetic, on the same texts and matched channels. Evaluate v1 and the new probes on the same cases. Record provider, model revision, locale, voice and acquisition date. The existing private examples are reserved for final evaluation.
4. **If the pilot passes**, run partial XLS-R adaptation, distil, export, and evaluate the deployed int8 model. Record validation false accusations and false reassurances, abstention, teacher/student difference and target-laptop latency before freezing a release. A 2% false-alarm budget needs independent speakers and denominators.

## Verification and ownership

The combined backend and pipeline suite passed all **41 tests** (five warnings). The 12 v3 tests cover source matching, speaker/voice/text/parent leakage, held-family checks, shortcut controls, shard corruption/recovery, in-memory container decoding and the quantized PRD adapter. Ruff and Python compilation passed. Notebook cells were syntax-checked locally. External synthesis, authenticated dataset joins, cloud acceleration and an end-to-end trained export have not been exercised. PyTorch emits deprecation warnings for the dynamic quantization/TorchScript path, which passed the local CPU fixture.

Preserve pre-existing work in `.gitignore`, the x1 changes under `detector/`, `detector/datagen/`, `detector/eval/compare_dev.py`, `google_collab/` and the owner's prompt. The research implementation lives in `pipeline_v3/`; app integration lives in `app/`. Do not write to the frozen pipelines, push the branch, or commit audio, weights, datasets, keys or papers. The owner controls cloud spending, service credentials and voice-cloning consent. All sealed sets, including `app_testing_dataset/`, stay outside development.
