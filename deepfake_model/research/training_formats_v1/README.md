# Native versus mixed-format CNN–LSTM training

On the same 734 native 16 kHz mono development recordings, native-only training detected 63.8% of fakes on average, compared with 53.9% for the matched mixed-format control. Every comparison allowed the same eight false alarms among 167 genuine recordings (4.79%). The native arm won two of three paired seeds. Its average AUC advantage was small (0.8483 versus 0.8417).

| Training seed | Native detected / 567 | Mixed detected / 567 | Native AUC | Mixed AUC | Native training | Mixed training |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 349 | 228 | 0.8160 | 0.7956 | 13.85 s | 12.66 s |
| 43 | 337 | 348 | 0.8304 | 0.8323 | 12.69 s | 12.57 s |
| 44 | 400 | 341 | 0.8984 | 0.8971 | 12.58 s | 12.66 s |

For context, the existing full-data baseline scored 301/567 (53.1%) at the same empirical false-alarm count, using its previously saved dev scores. This comparison uses a separate cutoff for each model, computed on these genuine dev recordings. The cutoffs describe the ranking at a common operating point and have not been independently calibrated for deployment.

**Experimental controls.** Each arm used 997 genuine and 1,338 fake recordings from the same nine train speaker folders. The mixed arm sampled once within each speaker/class stratum from all available source formats. It shared 1,851 native files with the native arm, replacing 484 with files of other formats. We matched the number of eligible one-second windows per paired recording (12,984 per epoch per arm). Both arms used the existing model, mean recording score, AdamW, batch size 96, BF16 autocast and the final epoch of a fixed five-epoch schedule. Seeds 42, 43 and 44 varied initialization and shuffling on fixed cohorts. Neither ASVspoof nor reserved test audio was scored or used for selection.

**Measured timing.** On the local RTX 5060, the median five-epoch fit was 12.66 seconds (about 2.53 seconds/epoch). Timers synchronized CUDA at epoch boundaries and included the loader and row-ID checks. Building the dev and two training feature arrays took 23.77 seconds in total, shared across seeds. The complete six-model experiment took 106.39 seconds from the start of the experiment driver through prediction and artifact writing. Python import/startup and the independent post-run audit are outside that interval.

**Interpretation.** The dev sample covers three speaker folders, with genuine examples in only two. For Speaker-50, native training detected between 4 and 63 of 154 fakes across seeds; the mixed condition detected between 0 and 45. The results support carrying native-format training forward as a candidate. Evidence for use on calls requires an independent calibration set and fresh speaker/source-disjoint recordings with matched channels and both labels. Neither the dev operating point nor the best seed should be treated as a production model.

**Verification.** Both arms' recorded row IDs were checked against the baseline train manifest, including counts by speaker/class. All six runs produced one score per dev file. A separate pass reconstructed AUC and confusion counts from those scores and checked that the logged epoch times sum to each training duration.

[Per-speaker and overall metrics](metrics.csv) · [Recording scores](dev_predictions.csv) · [Training membership and retained row IDs](membership.csv) · [Per-epoch timings](timing.csv) · [End-to-end timing](elapsed.json) · [Configuration and hashes](config.json)

Reproduce from the repository root with a new output directory:
```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python deepfake_model/pipeline/compare_training_formats.py --device cuda --out deepfake_model/research/training_formats_repeat
```

The retained candidate weights are in [native_42/checkpoint.pth](native_42/checkpoint.pth). The other five comparison runs retain their membership, predictions, metrics and timings. Their weights can be regenerated with the command above.
