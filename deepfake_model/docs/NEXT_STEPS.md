# Development plan and known issues

## 1. Restore a reproducible baseline

- Create a fresh environment, validate imports and record tested package versions.
- Smoke-test checkpoint loading and file inference without overwriting historical outputs.
- Keep the copied features and results as historical artifacts; write corrected runs to new output directories.

## 2. Correct training and evaluation before reporting performance

- Fix `pipeline/02_train_cnn_lstm.py`: evaluation currently attaches a sequential sampler over a subset to the full dataset. The emitted indices select the first N samples, not the held-out sample IDs. Use a DataLoader over the actual Subset.
- Select thresholds and fit calibration using validation data only. Current outer-test threshold selection makes threshold-dependent evaluation optimistic.
- Associate calibration with the selected inner model's genuinely held-out validation partition. The current code always uses the last inner validation partition.
- Select the deployment model without choosing by outer-test AUC. Reserve an untouched final test set.
- Check actual speaker identity and duplicate/source overlap, particularly within `Unspecified_Audio_Samples`; directory groups alone do not establish speaker independence.
- Compute class weights from each training partition and step OneCycleLR at its intended batch frequency.
- Decide whether to model temporal order: the present LSTM sequence axis is frequency, not time. Changing this requires a newly trained model.
- Align feature extraction for training files and inference chunks, including normalization, padding and chunk boundaries.

Recorded AUC 0.9630 and the 10.14x instability ratio are historical values, not verified speaker-independent results after these corrections.

## 3. Validate detection on realistic audio

- Use separate development and final evaluation sets containing unseen speakers and synthesis methods.
- Evaluate noise, compression, silence, short utterances and call-channel effects.
- Report fake recall, real specificity, false alarms, AUC, calibration and latency alongside accuracy.
- Existing VA results contain only 44 files. The original CSV labels all 44 real; v2 correctly identifies 22 fake and 19 real files, with 3 false positives. These files have already informed development and should not become the sole final test set.
- Compare threshold strategies under the same validation protocol; their scores are correlated and should not be treated as independent evidence.

## 4. Unify streaming inference

- Consolidate duplicated model, feature and stabilizer implementations.
- Resolve the unused voting decision in `process_chunk`: the final label currently comes from EMA confidence instead.
- Save the calibration parameters and corresponding thresholds needed at inference.
- Make file and microphone verdict rules consistent. Currently file mode uses a 15% fake-chunk or 0.90 max-score trigger; microphone mode uses 40% fake chunks.
- Handle incomplete final chunks, insufficient speech, silence and uncertain decisions explicitly.
- Benchmark CPU inference time and end-to-end streaming delay before choosing a model for the app.

## 5. Build the app

Choose the target platform and audio source first: an in-app VoIP call, desktop call-audio capture, or another supported integration. Verify that platform's audio access before promising detection during ordinary phone calls.

Suggested implementation sequence:

1. File-upload demo connected to the validated model.
2. Live microphone streaming with a continuously updated assessment and an insufficient-audio state.
3. Integration with the chosen call-audio source.
4. Measure detection delay, false alarms, compute use and behavior across devices.

Current preparation creates the workspace only; it does not implement these fixes or the app.
