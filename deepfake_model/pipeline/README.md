# Speaker holdout baseline

The active entry points are 01 (manifest and historical crosswalk), 02 (training and one-shot evaluation) and 04 (file, folder and microphone inference). See [the evaluation report](../REPORT.md) for results and exclusions.

From the repository root, use the project's Python environment. The retained run was made with Python 3.14.7 on an RTX 5060, with the versions in `deepfake_model/pipeline/outputs/audited_v1/full/environment.txt`. The local raw corpus and original feature arrays are required to reproduce the historical audit. They are outside Git. Manifest paths are relative to `deepfake_model/datasets/indiefake/`.

```bash
source venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
p=deepfake_model/pipeline
r="$p/outputs/reproduction_01"
python "$p/01_prepare_dataset.py" --out "$r"
python "$p/02_train_cnn_lstm.py" --manifest "$r/manifest.csv" --out "$r/smoke" --smoke --epochs 2 --device cuda
python "$p/test_baseline.py" --manifest "$r/manifest.csv" --checkpoint "$r/smoke/checkpoint.pth"
python "$p/02_train_cnn_lstm.py" --manifest "$r/manifest.csv" --out "$r/full" --epochs 5 --device cuda
python "$p/measure_external.py" --run "$r/full"
```

Reproduction of the fixed run reuses an already reported test set. Use fresh speakers for evaluating changes informed by the results. Directories for preparation and training must be new, and CSV writers refuse to overwrite files. The feature loader decodes audio in memory. Each nonconstant one-second chunk, including a zero-padded partial last chunk, contributes one equally weighted score. Constant feature matrices contribute no evidence. Empty or entirely constant streams return `INSUFFICIENT`.

To run the retained checkpoint, use `python deepfake_model/pipeline/04_realtime_inference.py --mode file --input_file /path/to/audio.wav`. Microphone mode takes `--mode mic --duration 30` and requires sounddevice, PortAudio and a 16 kHz input. External AAC recordings require ffmpeg. An explicit baseline checkpoint requires its matching `rule.json` in the same directory.
