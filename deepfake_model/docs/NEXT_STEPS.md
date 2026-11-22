# After the audited baseline

The baseline has a frozen CNN–LSTM checkpoint, a dev-selected recording rule, a one-time evaluation on ten reserved speaker folders and a replay test for file and microphone decisions. See [REPORT.md](../REPORT.md) and [the reproduction commands](../pipeline/README.md).

The [paired call-channel experiment](CALL_CHANNEL_PROBE.md) found that an 8 kHz linear PCM round trip flipped seven of nine genuine excerpts to FAKE. Prioritize genuine/fake recordings captured through the same channel. Use the external provenance audit to remove training copies before constructing any new development cohort.

The [completed ASVspoof evaluation and source-format audit](../research/asvspoof_la_v1/README.md) found high false alarms even on uncompressed speech. The [matched training experiment](../research/training_formats_v1/README.md) found mean fake recall of 63.8% for native 16 kHz mono training and 53.9% for mixed-format training at eight dev false alarms each. Keep native-format training as a candidate and calibrate on independent speakers before assessing fresh call recordings.

1. Collect a call-audio development set with known speaker and source identities, matched real/fake durations, multiple devices, call codecs and synthesis methods. Annotate speech intervals and synthetic segment onsets so false-alert exposure and detection delay can be measured directly.
2. Reserve a separate speaker/source-disjoint confirmation set before experimenting. The ten reported test speaker folders and the 44 VA recordings already have published results in this workspace.
3. On development data, measure alerts per genuine hour, missed fake calls and detection delays at a false-alert budget chosen for the intended use. Check whether the model relies on channel, source or duration cues. Compare changes to this frozen baseline.
4. Decide on architecture and live-call integration after measuring transfer to calls. Hardware microphone capture, remote-party capture and end-to-end device latency still require separate testing.
