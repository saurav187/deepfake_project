# Frozen detector on ASVspoof 2021 LA

The detector falsely flagged **63.1% of genuine uncompressed utterances** and **67.8% of genuine µ-law utterances**. Both rates are too high for a caller-facing warning. The µ-law condition also detected a larger fraction of spoofed utterances. Performance on uncompressed speech shows that addressing the telephone codec alone would leave a substantial problem.

| Official codec | Genuine flagged / scored | Spoofs detected / scored | Recording AUC | Unverified genuine / spoof |
|---|---:|---:|---:|---:|
| none | 1,107 / 1,753 (63.1%) | 10,458 / 15,449 (67.7%) | 0.6224 | 196 / 2,023 |
| ulaw | 1,469 / 2,168 (67.8%) | 15,539 / 19,412 (80.0%) | 0.6975 | 188 / 1,752 |

The archive matched the publisher's checksum. The run visited 181,566 FLAC entries and covered all 42,941 unique CM protocol IDs with phase `eval`, trimming `notrim` and either selected codec. It scored 38,782 utterances. For 4,159 (9.7%), decoding or PCM verification failed; the prediction CSV retains those IDs as errors. Rates and AUC use the verified utterances. Even counting every unverified genuine file as correctly accepted would leave false-positive rates of 56.8% for none and 62.4% for µ-law. No scored file exactly matched a decoded recording hash in the IndieFake manifest.

The saved rule and checkpoint digest matched the audited baseline. All reported binary verdicts agreed with the threshold 0.5996407214552164, and all 498 pilot rows reproduced. The two codec groups are observational and include different transmission conditions. These short-utterance rates do not estimate false alerts per hour on a continuous call.

## Follow-up on the original training corpus

A lookup using only the source file's sample rate and channel count achieved **0.9545 dev AUC**. We fit each format's fraction of fake recordings using IndieFake train labels, then scored the separate dev speaker folders using those fractions. In training, every stereo source was genuine (2,678 recordings), and all 3,609 mono sources stored at 24 or 44.1 kHz were fake. Both labels occurred together only in the native 16 kHz mono group. [Source-format counts](source_formats.csv)

We reran the frozen CNN–LSTM on all 734 dev recordings in that shared format, verifying each eligible chunk against its baseline feature hash. Its AUC was **0.8294**, compared with 0.9726 over the entire dev set. It detected **265/567 fakes (46.7%)** and falsely flagged **1/167 genuine recordings (0.6%)** in this subset. The model receives standardized 16 kHz mono features, without file header fields. The lookup demonstrates a corpus confound; it cannot establish which waveform cues the model learned. The difference from ASVspoof also shows that matching the stored format leaves substantial differences between sources. [Within-format dev predictions](matched_format_dev.csv)

**Next experiment:** train a comparison copy of the same CNN–LSTM using the native 16 kHz mono train recordings from both classes (997 genuine, 1,338 fake). Compare it to the frozen baseline on exactly the same 734 dev recordings and report both fake recall and genuine false positives at comparable operating points. Use a size-matched mixed-format train control to distinguish the effect of format restriction from the loss of training examples. Keep the reported ASVspoof labels out of fitting and reserve new source-disjoint call recordings for confirmation.

## Reproduction and evidence

From the repository root, choose new output directories for either run:

```sh
OPENBLAS_NUM_THREADS=1 venv/bin/python deepfake_model/pipeline/evaluate_asvspoof_la.py --out /tmp/asvspoof-la-full-repeat
OPENBLAS_NUM_THREADS=1 venv/bin/python deepfake_model/pipeline/audit_source_formats.py --out /tmp/indiefake-format-audit
```

[Per-utterance scores and errors](predictions.csv) · [Coverage and confusion counts](summary.csv) · [Run hashes](run.json) · [Official audio and attribution](https://zenodo.org/records/4837263) · [Official protocol](https://www.asvspoof.org/index2021.html)
