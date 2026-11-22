Fix three issues found in review of the Deepfake Audio Check app in `app/`. The app was built from `docs/PRD_deepfake_audio_app.md`. Read that PRD (especially sections 7, 8 and 10), `AGENTS.md`, and the files named below before changing anything. All current checks pass; keep them passing.

## Constraints
- Change files only under `app/`. Do not modify `deepfake_model/`, `detector/`, `docs/` or `AGENTS.md`.
- Use the repo virtualenv `venv/`. Don't add dependencies.
- Uploaded audio must never be logged or written anywhere except a temp file deleted in a `finally` block.
- Keep the changes small and focused. No refactors beyond what each fix needs.
- Do not commit. Leave the changes in the working tree.

## Fix 1: the per-job log line never appears
`app/backend/jobs.py` logs `job_id`, `duration_s`, `verdict` and `elapsed_s` with `logging.getLogger(__name__)` (the `app.backend.jobs` logger). Under uvicorn that logger has no handler and an effective level of WARNING, so the INFO line is dropped. PRD section 10 requires these logs (job id, duration, verdict and timing only).

- Make the line appear when the app runs with `venv/bin/python -m uvicorn app.backend.main:app --no-access-log`. For example, log through a child of the `uvicorn.error` logger, which `main.py` already uses, or configure the `app.backend` logger once when the app is created. Don't call `logging.basicConfig` and don't change the root logger.
- Also log failed and cancelled jobs, with the job id, status and error code only.
- Add a test that captures logs while a job completes and asserts that the line contains the job id, the verdict and the duration. Keep the existing assertion that no audio bytes appear in the logs.

## Fix 2: detector load failures are logged without a reason
In `create_app`'s lifespan in `app/backend/main.py`, `except Exception:` logs only "The configured detector could not be loaded." That makes problems such as a missing checkpoint, a CUDA error or an import error impossible to diagnose.

- Log the exception type and message, and the traceback (`logger.exception`). No user audio exists at startup, so this is safe.
- Keep the current behavior otherwise: the server starts, `/api/health` reports `model_loaded: false`, and analysis returns `503 model_unavailable`.
- Add a test with a detector backend whose constructor raises (monkeypatch `load_backend`) that asserts the exception message appears in the log and that health reports `model_loaded: false`.

## Fix 3: the legacy adapter patches a module global for the whole process
`app/backend/detectors/legacy.py` sets `self.pipeline.read_audio.__globals__['subprocess'] = DecoderSubprocess`. That replaces `subprocess` inside the legacy `audio_features` module for the entire process. It works today, but it silently affects any other code that uses that module, and it breaks if the legacy module ever uses another `subprocess` function.

The legacy decoder being wrapped (`deepfake_model/pipeline/audio_features.py`, unchanged) is:

```python
def read_audio(path):
    try:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
    except sf.LibsndfileError:
        # External recordings include AAC in M4A containers.
        data = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                        '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'])
        return np.frombuffer(data, dtype='<f4')
    mono = audio.mean(axis=1)
    return librosa.resample(mono, orig_sr=rate, target_sr=SR) if rate != SR else mono
```

- Remove the global patch. Give the adapter its own decode function that reproduces those steps exactly: the same libsndfile read, channel mean and `librosa.resample` call for files libsndfile can read, and for everything else the same ffmpeg output arguments (`-f f32le -ac 1 -ar <SR>`). Keep the adapter's existing hardening on the ffmpeg fallback: the `INPUT_OPTIONS` protocol/format whitelist from `app/backend/media.py`, stderr sent to `DEVNULL`, and a timeout. Use `SR` from the legacy module rather than hard-coding 16000.
- The model must see exactly the same samples as before. Add a test that, for a WAV fixture at 44.1 kHz stereo, a 16 kHz mono WAV and an M4A fixture, asserts `np.array_equal(adapter_decode(path), legacy read_audio(path))`. Mark the test to skip when the legacy checkpoint or `deepfake_model/pipeline` is absent.
- Add a test that importing and constructing `LegacyDetector` leaves `audio_features.subprocess` as the real `subprocess` module. Skip it when the checkpoint is absent.

## Definition of done
1. `DETECTOR_BACKEND=mock venv/bin/python -m pytest app/backend -q` passes, including the new tests. Run the legacy tests too (the checkpoint exists on this machine) and report whether they ran or were skipped.
2. `venv/bin/python -m ruff check app/backend` passes.
3. The frontend is untouched; confirm `npm --prefix app/frontend run build` still succeeds.
4. Start `DETECTOR_BACKEND=legacy venv/bin/python -m uvicorn app.backend.main:app --port 8765 --no-access-log`, upload `deepfake_model/datasets/va_dataset/fake/tts_1.m4a` and `deepfake_model/datasets/va_dataset/real/voice_72.wav` through `/api/analyze`, and show:
   - the per-job log lines from the server output
   - each job's verdict and `fake_probability`. They must match the values before your change: `tts_1.m4a` gives `likely_fake` at 0.715 and `voice_72.wav` gives `likely_real` at 0.032 (rounded to 3 decimals). A difference means the decode changed; fix it rather than updating the expected values.
   Stop the server afterwards.
5. Start the server with `DETECTOR_BACKEND=detector` (whose module doesn't exist yet) and show the startup log, which should now include the reason the detector failed to load.

## Final report
List the files changed, the new tests, the outputs from checks 1–5, and anything you couldn't do.
