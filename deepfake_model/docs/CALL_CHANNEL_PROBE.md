# Call-channel sensitivity of the frozen detector

The 8 kHz round trip alone made the detector flag seven of nine genuine excerpts. Applying G.711 µ-law at the same sample rate raised that to eight. These are development observations from three source recordings. They explain why testing a call app requires genuine calls passed through the same channel as the fakes.

| Replay of the same ten-second excerpts | Fake excerpts flagged at end | Genuine excerpts flagged at end | Conversion scored above its paired original |
|---|---:|---:|---:|
| Original decoded signal | 0/45 | 0/9 | 40/45 |
| 8 kHz linear PCM, returned to 16 kHz | 42/45 | 7/9 | 26/45 |
| 8 kHz G.711 µ-law, returned to 16 kHz | 42/45 | 8/9 | 20/45 |

The median paired score difference (conversion minus original) moved from +0.084 to +0.025 with resampling and to −0.009 with µ-law. Loss of bandwidth was sufficient to cause most observed genuine threshold crossings. This also weakens a threshold-only explanation: after µ-law the paired score ordering was close to evenly split. These comparisons do not locate a specific frequency cue or establish behavior on captured calls.

## Which audio entered the experiment

The saved whole-file VA results had flagged 0/15 source-named conversions and 6/7 TTS files. Each conversion filename identifies an apparent source family (Linus, Obama or Ryan). For each family we paired its conversions with one genuine original by filename. We took the first, middle and last ten seconds at identical offsets over the family's shared duration. The offsets use whole seconds and never overlap. This gives 15 conversions × 3 excerpts and 3 originals × 3 excerpts. Several conversions share each genuine control. Source attribution is inferred from filenames; there are no annotations verifying identity, speech alignment or manipulation onset.

Every trial starts with fresh stream state, uses the existing inference path and ends at the frozen recording threshold of 0.5996407214552164. The linear PCM condition was added after the first original/µ-law run to test whether resampling alone could reproduce the effect. The 108 original and µ-law results reproduced exactly in the second run. Both channel paths retained 160,000 samples per excerpt, preserved a 600 Hz tone's level within 1%, and attenuated a 6 kHz tone to below 1% of input RMS. This simulation covers resampling and encoding; packet loss, microphones and network echo processing need recordings of real calls.

## External data provenance

The SHA-256 audit of decoded 16 kHz mono audio found that **9 of the 22 VA genuine files exactly match training recordings**. Another five match excluded IndieFake files from the unspecified speaker group. The remaining eight genuine files and all 22 fake files had no exact decoded-audio match. The three controls and all 15 conversions in this probe came from the unmatched files. Exact matching cannot rule out edited copies or shared speakers.

The nine training copies account for only 46.4 seconds of the original 1.065 genuine hours. After removing all 14 matching files, the previous zero-alert observation still covers 1.040 hours, concentrated in eight recordings. Interpret both exposure and the number of independent sources when using that earlier result.

## Reproduce

From the repository root run the following with the audited model, datasets and ffmpeg available. Choose a new output directory. Audio stays in memory.

```sh
OPENBLAS_NUM_THREADS=1 venv/bin/python deepfake_model/pipeline/probe_call_channel.py --out /tmp/call_channel_replay_v1
```

[Per-excerpt predictions](../research/call_channel_control_v1/trials.csv) · [Per-family comparisons](../research/call_channel_control_v1/comparison.csv) · [Full 44-file provenance audit](../research/call_channel_control_v1/corpus_audit.csv) · [Run hashes and versions](../research/call_channel_control_v1/run.json)
