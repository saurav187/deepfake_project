# Agent notes

- App work: follow `docs/PRD_deepfake_audio_app.md`. Put all app code under `app/`.
- Do not modify `deepfake_model/` (legacy research pipeline, frozen results) or `detector/` (new model pipeline, owned separately). Import from them only as the PRD's section 7 describes.
- Python environment: `venv/` at the repo root (Python 3.14). Do not upgrade or remove packages that `deepfake_model/requirements.txt` pins.
- Large data, model weights and run outputs are git-ignored. Never commit audio, `*.pth`, archives or anything under `*/datasets/` or `detector/runs/`.
- Never log or persist uploaded audio.
