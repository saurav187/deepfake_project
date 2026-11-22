# PRD: Deepfake Audio Check (upload-and-analyze web app)

| | |
|---|---|
| Status | Ready for implementation |
| Owner | Project owner (deepfake_project) |
| Implementer | Coding agent (Codex) |
| Model work | Done separately in `detector/` — **not** part of this PRD |
| Last updated | 2026-11-23 |

## 1. Summary

A web app where a user uploads an audio (or video) file and gets an assessment of whether the speech is AI-generated: a three-level verdict, a calibrated probability, and a timeline showing which parts of the recording look synthetic. The detection model is being developed in parallel. The app must therefore be built against a fixed **detector interface** (section 7), run first with a mock detector, then with an adapter for the existing experimental model, and later accept the new model with no app changes.

## 2. Problem and goals

People receive voice notes, clips and recordings and cannot tell whether a voice was cloned or synthesized. They need a fast, private, honest second opinion.

**Goals**
- G1. Upload a file and get a result in under ~15 s for a 1-minute file on a machine with an NVIDIA GPU (under ~60 s on CPU).
- G2. Make uncertainty visible. Never present a guess as a fact.
- G3. Show *where* in the recording the suspicious audio is.
- G4. Privacy by default: audio is never stored after analysis and never leaves the server.
- G5. Swap detection models without touching app code.

**Non-goals (out of scope for v1)**: live phone-call monitoring, user accounts or login, model training or evaluation, speaker identification ("is this really person X?"), mobile native apps, payments.

## 3. Users and use cases

| User | Scenario |
|---|---|
| Everyday user | Gets a WhatsApp voice note asking for money "from a relative"; uploads it to check. |
| Journalist / fact-checker | Checks a viral clip of a politician before reporting on it. |
| Student / researcher (project owner) | Demonstrates and inspects model behavior on files. |

## 4. User stories and requirements

Priority: **P0** = required for v1, **P1** = should have, **P2** = nice to have.

### Upload
- **P0** As a user, I can drag-and-drop or pick a file. Accepted: `.wav .mp3 .m4a .aac .ogg .opus .flac .webm .mp4 .mov` (for video files only the audio track is analyzed).
- **P0** Files over **50 MB** or longer than **10 minutes** are rejected with a clear message *before* analysis (size check in the browser and on the server; duration check on the server).
- **P0** Unsupported or corrupt files produce a clear, non-technical error ("We couldn't read this file as audio").
- **P1** I can record up to 60 s from my microphone in the browser and analyze it.

### Analysis
- **P0** After upload I see progress states: *Uploading → Analyzing → Done*. Analysis runs as a background job; the page polls for the result.
- **P0** I can cancel a pending analysis.

### Result
- **P0** A verdict card with one of four states (exact wording in section 9):
  `likely_real`, `uncertain`, `likely_fake`, `insufficient_audio`.
- **P0** The card shows the overall fake probability as a percentage and a one-sentence explanation of what it means.
- **P0** A timeline under a waveform of the recording, colored by segment fake probability (green → amber → red), with a legend.
- **P0** Clicking a segment plays the audio from that point. Playback uses the local browser copy of the file; nothing is re-downloaded.
- **P0** A permanent, visible caveat (section 9) and the model name and version.
- **P0** If the active model is marked `experimental` (see 7.3), show a prominent banner: "Experimental model — results on real-world audio are unreliable."
- **P1** A table of segments (start, end, probability) below the timeline.
- **P1** "Download report" exports the result as JSON (no audio inside).
- **P2** Session history: the last 10 results in this browser tab (stored in memory/sessionStorage only, never on the server).

### General
- **P0** Works on current Chrome, Firefox and Safari, desktop and phone width (≥ 360 px).
- **P0** Light and dark theme follow the OS setting.
- **P0** Keyboard accessible; the verdict is conveyed by text and icon, not color alone (WCAG 2.1 AA contrast).

## 5. User flow

```
[Landing: drop zone + short explanation + privacy note]
      │ select file
      ▼
[Client checks: type, size ≤ 50 MB] ──fail──► inline error
      │ POST /api/analyze
      ▼
[Progress: Uploading… → Analyzing… (Cancel)]  ◄── poll GET /api/jobs/{id} every 1 s
      │ status=done
      ▼
[Result: verdict card · probability · waveform+timeline · segments · caveat · model info]
      │ "Check another file"
      ▼
[Landing]
```

Wireframe of the result view:

```
┌──────────────────────────────────────────────────────────────┐
│  ⚠ LIKELY AI-GENERATED            Fake probability: 87%       │
│  Most of this recording has strong signs of synthetic speech.│
├──────────────────────────────────────────────────────────────┤
│  ▁▃▅▇▅▃▁▂▅▇▇▅▃▁▁▂▃▅▇▅▃▂▁   (waveform)                         │
│  ███████░░░░█████████████   (segments: green/amber/red)      │
│  0:00        0:15        0:30        0:45                    │
│  ▶ Play from selected segment                                │
├──────────────────────────────────────────────────────────────┤
│  This is an automated estimate, not proof. …                 │
│  Model: wavlm-aasist v1.0 · threshold set 2026-12-01          │
└──────────────────────────────────────────────────────────────┘
```

## 6. Architecture

```
browser (React + Vite + TypeScript)
   │  multipart upload / JSON polling
   ▼
FastAPI backend (Python)  ──►  job runner (in-process worker pool)
                                   │
                                   ▼
                         Detector interface (section 7)
                     ┌──────────────┼──────────────────┐
                 MockDetector   LegacyDetector     (future) NewDetector
                                (deepfake_model/)   (detector/inference.py)
```

- **Backend**: FastAPI, Python 3.12+ (the repo's venv uses 3.14). Single process. Analysis runs in a `concurrent.futures` worker pool of size `ANALYSIS_WORKERS` (default 1 on GPU, 2 on CPU). The model is loaded once at startup.
- **Job store**: in-memory dict with TTL (results expire after 60 minutes; a background task purges them). No database in v1.
- **Frontend**: React 18 + Vite + TypeScript. Waveform and timeline drawn with plain Canvas/SVG (no heavy charting or waveform libraries required). Audio is decoded for display in the browser with the Web Audio API.
- **Serving**: in production, FastAPI serves the built frontend from `app/frontend/dist` at `/`. In development, Vite proxies `/api` to the backend.
- **Packaging**: `Dockerfile` (P1) based on `python:3.12-slim` with `ffmpeg` installed; GPU optional.

### Repository layout (new code only under `app/`)

```
app/
  backend/
    main.py            FastAPI app, routes, static serving
    jobs.py            job store, worker pool, TTL purge, cancellation
    detectors/
      base.py          Detector protocol + result types (section 7)
      mock.py          MockDetector
      legacy.py        LegacyDetector (wraps deepfake_model/ CNN-LSTM)
      registry.py      picks the backend from DETECTOR_BACKEND
    config.py          settings from environment variables
    tests/
  frontend/
    src/ (components: DropZone, ProgressView, VerdictCard, Timeline, SegmentTable, Caveat, ModelBadge)
  README.md            how to run dev, run tests, build, deploy
```

**Do not modify** anything in `deepfake_model/` or `detector/`. Import from them only as described in section 7.

## 7. Detector interface (the contract)

### 7.1 Types (`app/backend/detectors/base.py`)

```python
from typing import Literal, Protocol, TypedDict

Verdict = Literal["likely_real", "uncertain", "likely_fake", "insufficient_audio"]

class Segment(TypedDict):
    start_s: float
    end_s: float
    fake_probability: float        # 0..1, calibrated

class ModelInfo(TypedDict):
    name: str                      # e.g. "legacy-cnn-lstm"
    version: str
    experimental: bool             # True => UI shows the experimental banner
    description: str               # one sentence for the model badge tooltip

class AnalysisResult(TypedDict):
    verdict: Verdict
    fake_probability: float | None # None only when verdict == "insufficient_audio"
    segments: list[Segment]        # ordered by start_s; may overlap; may be empty
    duration_s: float              # decoded duration
    speech_s: float                # duration after silence trimming
    warnings: list[str]            # user-facing, e.g. "Very short recording"
    model: ModelInfo

class Detector(Protocol):
    info: ModelInfo
    def analyze(self, data: bytes, filename: str) -> AnalysisResult: ...
```

### 7.2 Rules every detector follows

1. **The detector owns audio decoding.** The app passes the raw uploaded bytes. The app must not resample, convert to mono, or re-encode the audio itself. (Lesson from model development: any difference between training-time and serving-time decoding changes results.)
2. `analyze` is synchronous and thread-safe for one call at a time per worker. It may take seconds.
3. It raises `ValueError` for undecodable input. The backend maps that to the user-facing "couldn't read" error.
4. Verdict thresholds come from the detector, not the app. The app only renders the verdict.

### 7.3 Backends to implement

| Backend | `DETECTOR_BACKEND` | Purpose |
|---|---|---|
| `MockDetector` | `mock` (default in tests) | Decodes with ffmpeg to get the duration, then returns deterministic fake segments (e.g. probability from a hash of the bytes) so the UI can show every verdict state. `experimental: True`, name `mock`. A query/env switch lets tests force each verdict. |
| `LegacyDetector` | `legacy` | Wraps the existing frozen CNN-LSTM. Use `deepfake_model/pipeline/04_realtime_inference.py`: `Detector(checkpoint)` and `Stream(detector)`. Feed the decoded audio; the per-chunk scores are one-second segments. Decode exactly as that module's `read_audio` does (write the bytes to a temp file and call `read_audio(path)`). Map the cumulative mean score to `fake_probability`. Verdict: `>= threshold` from `rule.json` → `likely_fake`; `< threshold − 0.2` → `likely_real`; otherwise `uncertain`; no scored chunks → `insufficient_audio`. Set `experimental: True` and the description "Research prototype; unreliable on real-world audio". |
| `NewDetector` | `detector` | **Not implemented by Codex.** The model team will add `detector/inference.py` exposing `load_detector() -> Detector` with this exact interface. The registry should import it lazily so the app runs before it exists, and show a clear startup error if it is selected but missing. |

## 8. API

All responses are JSON. Errors use `{"error": {"code": str, "message": str}}` with the HTTP status below.

| Method & path | Description |
|---|---|
| `POST /api/analyze` | `multipart/form-data` with field `file`. Validates extension, size (≤ `MAX_UPLOAD_MB`, default 50) and duration (≤ `MAX_DURATION_S`, default 600, checked by probing with ffprobe). Returns **202** `{"job_id": "...", "status": "queued"}`. Errors: 400 `unsupported_type`, 413 `too_large`, 422 `too_long` / `unreadable`, 503 `busy` when more than `MAX_QUEUE` (default 10) jobs are pending. |
| `GET /api/jobs/{job_id}` | `{"job_id", "status": "queued" \| "processing" \| "done" \| "failed" \| "cancelled", "progress": 0..1 \| null, "result": AnalysisResult \| null, "error": {...} \| null, "created_at", "expires_at"}`. 404 `not_found` for unknown or expired jobs. |
| `DELETE /api/jobs/{job_id}` | Cancel a queued job, or discard a finished one. 204. |
| `GET /api/model` | The active `ModelInfo` plus limits: `{"model": ModelInfo, "limits": {"max_upload_mb", "max_duration_s", "accepted_extensions"}}`. The frontend uses this for client-side checks and the model badge. |
| `GET /api/health` | `{"status": "ok", "model_loaded": bool}`. |

Notes:
- The upload is read into memory (bounded by the size limit) or a `tempfile.NamedTemporaryFile`, which is deleted in a `finally` block as soon as analysis ends. Raw audio is never written anywhere else and never logged.
- Rate limit (P1): 10 analyses per IP per 10 minutes, in memory.
- CORS: only the configured frontend origin in production.

## 9. Copy (exact user-facing text)

| Verdict | Title | Explanation |
|---|---|---|
| `likely_fake` | Likely AI-generated | Large parts of this recording show strong signs of synthetic speech. |
| `uncertain` | Can't tell | The signs are mixed. Treat this recording with caution and verify it another way. |
| `likely_real` | No strong signs of AI | We didn't find strong signs of synthetic speech. This doesn't prove the recording is genuine. |
| `insufficient_audio` | Not enough speech | We couldn't find enough clear speech to analyze. Try a longer or clearer recording. |

Permanent caveat under every result:
> This is an automated estimate, not proof. Detectors can miss new voice-cloning tools and can be wrong on heavily compressed, noisy or edited audio. Don't make important decisions based on this result alone — verify through another channel, such as calling the person back on a known number.

Privacy note on the landing page:
> Your file is analyzed on our server and deleted immediately afterwards. We don't keep or share your audio.

## 10. Non-functional requirements

| Area | Requirement |
|---|---|
| Performance | API responds to upload within 1 s of the upload finishing. With `MockDetector`, a 1-minute file completes in < 2 s. The timeline renders a 10-minute file without UI jank (segments ≤ 600). |
| Reliability | One failing analysis never crashes the server. Worker exceptions become `failed` jobs with a user-safe message. The server keeps working if the model fails to load (health shows `model_loaded: false`; analyze returns 503 `model_unavailable`). |
| Security | Validate the file by decoding/probing, not only by extension. Never pass user-controlled strings to a shell (use argument lists with `subprocess`). Enforce limits before decoding. Set the security headers `X-Content-Type-Options`, `Referrer-Policy` and a CSP that allows only self. |
| Privacy | No audio persistence, no audio in logs, no third-party calls or analytics. Logs contain job id, duration, verdict and timing only. |
| Configuration | Environment variables: `DETECTOR_BACKEND` (`mock`/`legacy`/`detector`), `MAX_UPLOAD_MB`, `MAX_DURATION_S`, `MAX_QUEUE`, `ANALYSIS_WORKERS`, `RESULT_TTL_S`, `ALLOWED_ORIGIN`, `DEVICE` (`cuda`/`cpu`/`auto`). |
| Dependencies | Backend: `fastapi`, `uvicorn`, `python-multipart`, `pydantic` (v2), plus the system `ffmpeg`. The legacy backend additionally needs the repo's existing requirements (`deepfake_model/requirements.txt`). Frontend: React, Vite, TypeScript; Vitest and Testing Library for tests. |

## 11. Testing and acceptance criteria

**Automated tests (must pass in CI and locally with `DETECTOR_BACKEND=mock`):**
- Backend (pytest + httpx TestClient):
  - Generate fixtures at test time with ffmpeg (a 3 s 440 Hz tone with noise in WAV, MP3, M4A, OGG/Opus, and an MP4 with an audio track; a 0.3 s file; a 12-minute file; a text file renamed `.wav`).
  - Each fixture produces the expected status code or verdict.
  - Job lifecycle: queued → processing → done; cancellation; TTL expiry; 503 when the queue is full.
  - The temp file is deleted after success *and* after failure (assert that the temp directory is empty).
  - No raw audio bytes appear in captured logs.
- Frontend (Vitest + Testing Library): DropZone rejects an oversized file client-side; VerdictCard renders all four verdicts with the exact copy from section 9; Timeline maps probabilities to the three color bands with correct `aria-label`s.
- P1: one Playwright end-to-end test: upload a fixture → result card visible.

**Acceptance criteria (v1 done when):**
1. `app/README.md` steps start the dev servers in under 5 commands, and `DETECTOR_BACKEND=mock` works without a GPU or any model files.
2. With `DETECTOR_BACKEND=legacy`, the sample file `deepfake_model/datasets/va_dataset/fake/tts_1.m4a` analyzes end to end and shows the experimental banner.
3. All P0 stories in section 4 are demonstrably working, and all tests pass.
4. Uploading a 60 MB file is rejected in the browser without a network request, and on the server if sent directly.
5. No file remains in the temp directory after any analysis.
6. Lighthouse accessibility score ≥ 90 on the landing and result views.

## 12. Milestones

| # | Deliverable |
|---|---|
| M1 | Backend skeleton: routes, job runner, limits, `MockDetector`, backend tests. |
| M2 | Frontend: upload → progress → result with all four verdict states against the mock. |
| M3 | `LegacyDetector` adapter, model badge and experimental banner, acceptance criterion 2. |
| M4 | Polish: timeline playback, segment table, JSON report, dark mode, accessibility pass, README. |
| M5 (P1) | Dockerfile, rate limiting, microphone recording, Playwright test. |

## 13. Future (not v1)
- Swap in the new detector (`DETECTOR_BACKEND=detector`) once `detector/inference.py` ships.
- Per-segment explanations (which frequency bands drove the score).
- Batch upload, shareable result links (requires a storage and privacy design), and browser-extension or messaging-app integrations.

## 14. Open questions for the owner
1. Will the app be public, or a demo for an academic project? (This affects rate limiting, hosting and the licence of the training data behind the final model.)
2. Which languages should the UI support? English is the default; is Hindi needed?
3. Hosting target: a local machine with the GPU, or a cloud VM?
