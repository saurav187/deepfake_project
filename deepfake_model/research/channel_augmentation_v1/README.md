# Call-channel augmentation of the native-format CNN–LSTM

Training through simulated 8 kHz call channels removed most of the channel-induced false alarms on IndieFake dev, but **it did not make the detector separate ASVspoof genuine from spoofed speech**. On dev audio passed through an 8 kHz µ-law round trip, models trained only on clean audio flagged on average **140 of 167 genuine recordings (84%)** at their clean-audio threshold. Augmented models flagged **14.7 (8.8%)**. On ASVspoof 2021 LA, mean AUC stayed between 0.61 and 0.65 in both arms. Augmentation lowered false alarms only by lowering detection by a similar amount. The channel was one problem; it is not the main obstacle to transfer.

## Design

Both arms train the existing CNN–LSTM on exactly the native 16 kHz mono train cohort of the [format comparison](../training_formats_v1/README.md): 997 genuine and 1,338 fake recordings, 12,984 windows, nine speaker folders. Model, optimizer, five epochs, batch order and step count are identical. The **clean** arm reproduces that experiment's native runs: the seed-42 checkpoint is byte-identical, and the dev scores of all three seeds agree within 7×10⁻⁶. In the **augmented** arm, each training window independently takes one of three versions in each epoch: original, 8 kHz linear PCM, or 8 kHz G.711 µ-law (mu = 255, 8-bit), each resampled back to 16 kHz. Each version was used for about a third of the windows. The whole recording passes through the channel before windowing, and genuine and fake recordings are treated identically, so the channel carries no label information.

Training uses soxr resampling with numpy companding. Evaluation uses ffmpeg, as in the [call-channel probe](../../docs/CALL_CHANNEL_PROBE.md), so dev is not scored with the exact transform seen in training. Both implementations preserve a 600 Hz tone and remove a 6 kHz tone.

Each model's **clean cutoff** allows at most 8 false alarms among the 167 genuine clean dev recordings (the 5% budget used before). The same cutoff is then applied under each channel and on ASVspoof. That shows what a deployed threshold would do when the channel changes. **Own cutoff** recomputes the 8-false-alarm point within each channel, measuring ranking alone. No ASVspoof label was used for training, threshold choice or selection.

## IndieFake dev: 734 native-format recordings (167 genuine, 567 fake), mean of 3 seeds

| Training | Dev channel | AUC | Genuine flagged at clean cutoff | Fakes detected at clean cutoff | Fakes detected at 8 false alarms |
|---|---|---:|---:|---:|---:|
| Clean | original | 0.848 | 8.0 / 167 | 63.8% | 63.8% |
| Clean | 8 kHz PCM | 0.794 | 114.7 / 167 | 91.4% | 43.0% |
| Clean | 8 kHz µ-law | 0.832 | 140.3 / 167 | 96.7% | 49.3% |
| Augmented | original | 0.830 | 8.0 / 167 | 55.4% | 55.4% |
| Augmented | 8 kHz PCM | 0.875 | 12.7 / 167 | 61.5% | 54.0% |
| Augmented | 8 kHz µ-law | 0.883 | 14.7 / 167 | 60.4% | 56.2% |

The frozen audited baseline behaved like the clean arm: 80 and 84 genuine recordings flagged under the two channels. Augmentation costs 8.4 points of clean-audio recall at the clean operating point. Under the channels it gains AUC and recall at matched false alarms in every seed. The high "detected at clean cutoff" figures for clean training under a channel are not useful detection: the models flag most inputs once they are band-limited. [Per-seed rows](dev_metrics.csv)

## ASVspoof 2021 LA eval: 42,941 trials, 38,782 decoded and verified

The cutoff for each model is its clean IndieFake dev cutoff. Values are the mean over seeds; [per-model rows](asvspoof_summary.csv) also include EER.

| Training | Codec | AUC (seed range) | Genuine flagged | Spoofs detected |
|---|---|---:|---:|---:|
| Clean | none | 0.630 (0.608–0.656) | 63.2% | 71.4% |
| Clean | µ-law | 0.651 (0.502–0.755) | 81.5% | 85.1% |
| Augmented | none | 0.639 (0.552–0.705) | 64.0% | 73.2% |
| Augmented | µ-law | 0.606 (0.494–0.688) | 43.2% | 58.5% |
| Audited baseline, frozen rule | none / µ-law | 0.622 / 0.698 | 63.2% / 68.0% | 67.7% / 80.1% |

Scores barely rank genuine above spoof in any arm, and they change mostly with the training seed. On µ-law trials, augmentation moved the operating point down (fewer genuine and fewer spoofed utterances flagged) without improving the ranking. The seed range spans chance in both arms, so no seed should be chosen on these labels. ASVspoof trials are short single utterances from a different corpus with 13 attack systems (A07–A19). These rates do not estimate alerts per hour of a call.

**Consistency checks.** The audited baseline rescored on the GPU reproduced its full ASVspoof AUC (0.6224 / 0.6976) and its scored and errored counts exactly. 1,108 and 1,474 false alarms at the frozen threshold compare with 1,107 and 1,469 from the earlier CPU run; the extra flags are scores at the threshold that shifted with GPU rounding. The first 2,048 archive members reproduced the [pilot](../asvspoof_la_pilot_v1/README.md) confusion counts exactly. Its clean dev scores match the saved within-format scores within 9×10⁻⁴.

## Interpretation

1. **Keep channel augmentation.** It is cheap, label-neutral and fixes a measured failure: a phone channel alone no longer flips most genuine IndieFake audio to FAKE. It should be in any future training recipe for call audio.
2. **It does not fix transfer.** On a second corpus with different speakers, recording conditions and synthesis systems, every CNN–LSTM trained on IndieFake ranks close to chance. This matches the [format audit](../asvspoof_la_v1/README.md): what the model learns from IndieFake is largely source- and corpus-specific. More tuning of this model on IndieFake will not answer the call-use question.
3. **Next experiment:** change what the model learns from, not its threshold. Two options can be tested with the current evaluation unchanged:
   - **Training diversity.** Add the ASVspoof 2019 LA train partition (speakers and attacks disjoint from 2021 LA eval) to IndieFake. Use the same channel augmentation and select on held-out 2019 LA dev and IndieFake dev, never on 2021 LA eval.
   - **Pretrained front end.** Replace the log-Mel CNN with a frozen self-supervised speech encoder (for example wav2vec 2.0 / XLS-R or WavLM) and a small classifier head. These representations are the strongest published approach to cross-corpus anti-spoofing.

   Report both on 2021 LA eval at the IndieFake dev cutoff, and keep fresh call recordings for confirmation.

## Reproduce

From `deepfake_model/pipeline`, with a new output directory (training and dev scoring take about 2.5 minutes on an RTX 5060; the ASVspoof pass takes about 4 minutes):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../../venv/bin/python channel_augmentation.py --out ../research/channel_augmentation_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../../venv/bin/python channel_augmentation.py --asvspoof --out ../research/channel_augmentation_repeat
```

`--asvspoof --limit 2048` runs a smoke test on the first 2,048 archive members. Weights for all six models are stored locally in the `*_<seed>/` folders and are excluded from Git.

[Dev metrics](dev_metrics.csv) · [Dev recording scores](dev_predictions.csv) · [ASVspoof summary](asvspoof_summary.csv) · [ASVspoof per-utterance scores](asvspoof_predictions.csv) · [Configuration, checks and hashes](config.json) · [Training channel usage and timing](timing.csv) · [ASVspoof run](asvspoof_run.json)
