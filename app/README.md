# Deepfake Audio Check

Upload an audio or video recording to see an automated assessment of synthetic speech, a timeline and a JSON report. The server keeps jobs in memory and deletes its temporary audio files after use. Playback and waveform rendering use the original file in the browser.

## Development

Use the repository's Python 3.14 virtual environment, Node 26 with npm, and system `ffmpeg` and `ffprobe`. Run these commands from the repository root. Start the last two in separate terminals.

```sh
venv/bin/python -m pip install -r app/backend/requirements.txt
npm --prefix app/frontend ci
DETECTOR_BACKEND=mock venv/bin/python -m uvicorn app.backend.main:app --port 8000 --no-access-log
npm --prefix app/frontend run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8000. To demonstrate a particular mock verdict, set `MOCK_VERDICT` on the backend process to `likely_real`, `uncertain`, `likely_fake`, or `insufficient_audio`.

## Production and legacy model

Build the frontend and start one Uvicorn process from the repository root. FastAPI serves the build at <http://localhost:8000> and the API under `/api`.

```sh
npm --prefix app/frontend run build
DETECTOR_BACKEND=mock venv/bin/python -m uvicorn app.backend.main:app --port 8000 --no-access-log
```

Use `DETECTOR_BACKEND=legacy` in the Uvicorn command to load the frozen CNN-LSTM. It requires the preinstalled packages pinned by the research pipeline and its audited `checkpoint.pth` and `rule.json`. On a machine with a GPU, the legacy detector selects CUDA automatically. Set `DEVICE=cpu` to use the CPU. Try the supplied `deepfake_model/datasets/va_dataset/fake/tts_1.m4a` sample.

A deployment can run the production command behind an HTTPS reverse proxy. Use one Uvicorn worker so upload and polling requests share the same job store. If a reverse proxy is used, allow the configured file limit plus multipart overhead. Set `ALLOWED_ORIGIN` to a single origin when the frontend is served separately. The default is same-origin.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DETECTOR_BACKEND` | `mock` | `mock`, `legacy`, or `detector` when the new model ships |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `MAX_UPLOAD_MB` | `50` | Maximum file size in decimal MB |
| `MAX_DURATION_S` | `600` | Maximum recording duration in seconds |
| `ANALYSIS_WORKERS` | GPU 1, CPU 2 | Analysis pool size |
| `MAX_QUEUE` | `10` | Maximum jobs waiting for a worker |
| `RESULT_TTL_S` | `3600` | Retention in seconds once a job finishes or is cancelled |
| `ALLOWED_ORIGIN` | Unset | Explicit cross-origin frontend URL |

If startup cannot load the selected detector, `GET /api/health` reports `model_loaded: false`, the startup log reports the problem, and analysis returns `503 model_unavailable`.

## Tests and checks

```sh
DETECTOR_BACKEND=mock venv/bin/python -m pytest app/backend -q
venv/bin/python -m ruff check app/backend
(cd app/frontend && npm test -- --run)
npm --prefix app/frontend run check
npm --prefix app/frontend run build
```

The backend tests generate audio with system ffmpeg and exercise the API through FastAPI's TestClient. The frontend tests use Vitest and Testing Library.

## Decisions

- A megabyte means 1,000,000 bytes. The backend enforces the file size while parsing the multipart body in memory and bounds multipart overhead separately.
- ffprobe measures duration for server validation and for the mock's duration-only implementation. The adapters receive the original bytes. The frozen decoder's ffmpeg fallback is restricted to accepted media formats and its diagnostics are suppressed. Its decoding parameters are unchanged.
- Queue capacity counts waiting jobs. Cancelling a queued job releases its bytes immediately. Cancelling a running job discards the eventual result. The synchronous detector call finishes and cleans up its temporary input.
- Result expiry starts when a job reaches a terminal state. Jobs disappear when the server restarts.
- The mock uses a content hash to generate scores. Unless a verdict is forced, recordings shorter than one second return `insufficient_audio`. For the legacy model, speech duration is the total length of scored chunks and the reported version comes from the audited rule.
- The display uses green below 33%, amber from 33% to below 67%, and red from 67%. These bands are independent of the detector's verdict thresholds. Segment selection seeks local playback. Waveform and playback availability depend on the browser's support for the original codec. The result and report stay available if decoding fails on the device.
- The JSON download contains the API analysis result, including model metadata and segments.
