# Audio deepfake detection

This repository contains a speaker-independent CNN–LSTM research detector and the experiments evaluating its use on call audio. The current baseline, external results, and development priorities are documented in the [evaluation report](deepfake_model/REPORT.md) and [next steps](deepfake_model/docs/NEXT_STEPS.md).

| Retained model | Purpose |
| --- | --- |
| [Audited baseline](deepfake_model/pipeline/outputs/audited_v1/full/checkpoint.pth) | Reference detector with its [frozen decision rule](deepfake_model/pipeline/outputs/audited_v1/full/rule.json). Used by the file and microphone entry point. |
| [Native seed-42 candidate](deepfake_model/research/training_formats_v1/native_42/checkpoint.pth) | Experimental model trained on native 16 kHz mono recordings. It still needs calibration on separate call recordings. |

From the repository root, run the audited baseline on an audio file:

```bash
venv/bin/python deepfake_model/pipeline/04_realtime_inference.py --mode file --input_file /path/to/recording.wav
```

On a new machine, create the environment with `python3 -m venv venv` and install dependencies with `venv/bin/python -m pip install -r deepfake_model/requirements.txt`. See the [pipeline guide](deepfake_model/pipeline/README.md) for reproduction commands. The [matched training comparison](deepfake_model/research/training_formats_v1/README.md) and [ASVspoof evaluation](deepfake_model/research/asvspoof_la_v1/README.md) include per-recording results and timing or coverage. The large datasets, original feature arrays, and model weights are local and excluded from Git. When moving the project, copy those artifacts separately.
