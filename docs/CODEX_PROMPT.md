Build the "Deepfake Audio Check" web app in the empty `app/` folder of this repository, following `docs/PRD_deepfake_audio_app.md` exactly. Read the whole PRD and `AGENTS.md` before writing any code. The PRD is the source of truth; if this prompt and the PRD disagree, follow the PRD and tell me.

## Scope for this task
Implement milestones M1–M4 from PRD section 12 (all P0 stories, plus the P1 segment table and JSON report download). Do not do M5 (Dockerfile, rate limiting, microphone recording, Playwright) unless everything else is finished and passing.

## Hard constraints
- Write code only under `app/`. Do not modify `deepfake_model/`, `detector/`, `docs/` or `AGENTS.md`, and do not move or delete any data or model files.
- Python: use the existing repo virtualenv `venv/` (Python 3.14, with torch already installed; the legacy model needs it). Put backend dependencies in `app/backend/requirements.txt` and install them with `venv/bin/python -m pip install -r app/backend/requirements.txt`. Do not upgrade, downgrade or remove packages pinned in `deepfake_model/requirements.txt`. If a dependency conflict appears, stop and report it instead of forcing it.
- Frontend: React + Vite + TypeScript in `app/frontend/`, with npm. Node 26 and npm 12 are installed. Keep dependencies minimal: no UI kit, charting or waveform libraries; draw the waveform and timeline with Canvas or SVG.
- ffmpeg/ffprobe are installed system-wide. Always call them with argument lists, never a shell string.
- Never write uploaded audio anywhere except a `tempfile` that is deleted in a `finally` block. Never log audio bytes or file contents.
- The detector owns decoding (PRD 7.2): the backend passes raw uploaded bytes to `Detector.analyze`. The app must not resample or convert audio itself, except that `MockDetector` may use ffprobe to get the duration.
- Use the exact user-facing copy in PRD section 9.

## Detector backends (PRD 7.3)
- `MockDetector` (`DETECTOR_BACKEND=mock`, the default): deterministic, no GPU or model files needed. Provide a way for tests to force each of the four verdicts.
- `LegacyDetector` (`DETECTOR_BACKEND=legacy`): wrap `deepfake_model/pipeline/04_realtime_inference.py` (`Detector`, `Stream`, `read_audio`). The checkpoint and `rule.json` are in `deepfake_model/pipeline/outputs/audited_v1/full/`. Add `deepfake_model/pipeline` to `sys.path` inside the adapter only; don't copy the legacy code. Load the model once at startup. Mark it `experimental: True`.
- `detector` backend: a lazy import of `detector.inference.load_detector`, which doesn't exist yet. Selecting it must give a clear startup error, not a crash at import time.

## Definition of done (verify each yourself before finishing)
1. `DETECTOR_BACKEND=mock venv/bin/python -m pytest app/backend -q` passes. The tests generate their audio fixtures with ffmpeg at test time (see PRD 11) and cover every listed case, including temp-file cleanup after success and failure, and that no audio bytes appear in logs.
2. `cd app/frontend && npm test -- --run` passes, and `npm run build` succeeds with no TypeScript errors.
3. Dev mode works with at most 5 commands, documented in `app/README.md`: backend on port 8000 with uvicorn, Vite on 5173 proxying `/api`. A production mode in which FastAPI serves `app/frontend/dist` also works.
4. With `DETECTOR_BACKEND=legacy`, uploading `deepfake_model/datasets/va_dataset/fake/tts_1.m4a` through the API returns a `done` job with segments and the experimental flag. Show me the JSON you got.
5. A 60 MB upload is rejected in the browser (no request sent) and by the server (413) if posted directly.
6. Run the type checker or linter you set up (ruff for Python, `tsc --noEmit` for TypeScript) with no errors.

## How to work
- Build in milestone order (M1 → M4) and run the tests after each milestone.
- Make small, readable modules; follow the file layout in PRD section 6.
- If something in the PRD is ambiguous, pick the simplest reasonable option, note it in `app/README.md` under "Decisions", and continue. Stop and ask only for problems you can't resolve (for example dependency conflicts or missing system tools).
- Do not commit or push. Leave changes in the working tree.

## Final report
When finished, give me:
- what was built, per milestone
- the exact commands to run the app and the tests
- the test output summary
- the JSON from check 4
- any PRD requirement not met, and why
- the decisions you made on ambiguous points
