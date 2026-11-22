# ASVspoof 2021 LA transfer pilot

Using the unchanged audited checkpoint and its 0.5996407214552164 threshold, we scored the eligible trials among the first 2,048 FLAC members of the official archive. The selection used the countermeasure protocol, `eval` phase, `notrim` and either `ulaw` or `none`. The prefix is a transfer and pipeline check. It is not a random or speaker-balanced sample. The conditions are descriptive groups, not matched pairs. No ASVspoof labels were used to change the model or threshold.

| Condition | Genuine flagged | Spoofs detected | Genuine / spoof decode errors |
|---|---:|---:|---:|
| No codec | 10 / 18 (55.6%) | 119 / 174 (68.4%) | 1 / 30 |
| µ-law | 19 / 28 (67.9%) | 182 / 225 (80.9%) | 2 / 20 |

Of 498 eligible trials, 445 passed the sample-count and embedded PCM MD5 checks. Every scored recording was 16 kHz, mono and decoded to 16-bit PCM before conversion to float32. The [complete prediction table](../asvspoof_la_v1/predictions.csv) includes every pilot row and its decode status. Some streams that libsndfile rejects decode successfully in ffmpeg; both paths must match the same embedded checksum. For 53 files neither path yielded verified audio. The displayed rates use only verified audio. In particular, the small genuine denominators and decoding exclusions limit interpretation. No eligible verified recording matched a decoded-audio hash in the IndieFake manifest.

The full acquisition uses the [official audio archive](https://zenodo.org/records/4837263) and [ASVspoof challenge keys](https://www.asvspoof.org/index2021.html). The ASVspoof consortium distributes the audio under ODC-By. The evaluation script for the complete archive checks the published MD5 before scoring. It writes a separate result directory, preserving these pilot outputs.

The full comparison covers 42,941 eligible trial IDs, including all published evaluation transmissions within each selected codec. Short utterance false-positive rates here are not false alerts per hour of a continuous call.

From the repository root, the local acquisition and evaluation can be resumed with:

```sh
bash deepfake_model/pipeline/fetch_asvspoof_la.sh
```

The full evaluation is complete. See the [full summary and coverage](../asvspoof_la_v1/README.md).
