# Audited CNN–LSTM baseline, v1

On 5,824 IndieFake recordings from ten reserved speaker folders, the frozen detector achieved recording-level AUC **0.9939** and fake recall **94.48%**. On the separate 44-file VA development set it flagged **6 of 22 fake recordings** at file end. Invest next in matched call-channel data with verified identities, speech intervals and manipulation timestamps. The transfer gap is large enough that another round of tuning on IndieFake would answer the wrong question.

## Findings and repairs

- The historical fold-2 loader returned the first 2,321 rows. Only 140 belonged to its requested test fold. The training code also used outer-test labels for thresholds, selected a checkpoint by outer-test AUC and calibrated on the last inner partition irrespective of the selected model. The historical 0.963 mean AUC cannot estimate unseen-speaker performance.
- All 19,146 historical inputs were cross-referenced against freshly decoded audio using speaker folder, class, a candidate signature and a full-matrix numerical check (maximum absolute difference below 0.06 dB). Their 411 duplicate sets contained 847 rows. Of these, 117 sets (259 rows) crossed speaker folders. The raw-source audit classified 641 rows as sharing full decoded audio with a counterpart, 188 as sharing the decoded opening and 18 as having an identical floor-valued historical feature. All 18 conflicting-label recordings contained nonzero signal, but their openings peaked at no more than 0.000105 full scale. Whole-file normalization, the −80 dB floor and cropping collapsed the openings to identical features. The evidence points to silent/near-silent openings rather than full-scale waveform clipping.
- The historical extractor discarded source names. For 616 rows there are multiple interchangeable filenames even within the same feature hash, label and group. The historical crosswalk assigns these in sorted order and marks them ambiguous. The new chunk manifest is built directly from sorted source paths, so every new row has an observed source identity. Of 19,240 raw source paths, 94 yielded no decoded samples and are listed in the audit.
- Matplotlib rejected the historical boxplot keyword. The new run saved its checkpoint ahead of plotting and generated the plot with `tick_labels`. In the old external results, 19 of the 22 correctly flagged fake files and all three false positives qualified only by the peak rule. The old microphone rule used a different threshold on chunk counts.

## Split and frozen protocol

We shuffled 49 named folders with NumPy seed 42 and reserved ten each for dev and test. We excluded the unspecified group. We excluded whole recordings if decoded audio, historical inputs or informative chunk hashes were shared across speakers or labels. Within a group and label, we kept one copy of duplicate audio or duplicate historical inputs. Constant chunks contributed no evidence. A recording with no usable chunks cannot receive a binary verdict. Split assignment uses directory identity. Cross-folder voice identity and near-duplicate or shared generator/source identity remain unverified.

| Eligible split | Speaker folders | Recordings (real / fake) | Scored or trained chunks |
|---|---:|---:|---:|
| Train | 29 | 3,675 / 4,947 | 45,719 |
| Dev | 10 | 385 / 2,702 | 17,247 |
| Test | 10 | 3,016 / 2,808 | 29,605 |

The original three convolution blocks and bidirectional LSTM operate on 128 × 94 log-Mel inputs. The LSTM still sequences frequency bins. All paths now use the same 16 kHz mono, one-second windows, per-window normalization and zero-padding of the last partial window. Each eligible chunk receives equal weight. The CUDA backward pass for the original adaptive average pool failed PyTorch's strict determinism check. An equivalent fixed average for the CNN's 16 × 11 output passed CPU forward and gradient comparisons. The full run used deterministic PyTorch operations, BF16 GPU autocast for training, AdamW, five epochs and class weights from train only.

The two-epoch smoke run used 386 chunks from train folders, including its mock dev and mock test. Its fixture was identical in the preliminary and final manifests. The full run selected **epoch 2** by highest dev recording AUC using mean scores. At that epoch, we compared the mean chunk score with the fraction of chunks scoring at least 0.5, choosing the rule and threshold that maximized dev sensitivity plus specificity minus one. The winner was the **cumulative arithmetic mean ≥ 0.5996407214552164**. Each additional second contributes another observation without giving a single isolated peak a permanent veto. A sustained change late in a long recording can take longer to affect the cumulative mean.

The checkpoint and rule were frozen and hashed before test feature loading. Test was scored once. Subsequent checks replayed the saved scores arithmetically and used training audio for the live/file regression. Every loader pass checked actual IDs against requested IDs, including shuffled training passes. The manifest audit checked speaker, file, decoded audio, historical feature and new feature hashes across all active splits.

## Reserved test results

| Recording-level measure | Estimate | 95% speaker-bootstrap interval |
|---|---:|---:|
| AUC | 0.9939 | 0.9709–0.9997 |
| EER | 4.14% | 0.32–9.90% |
| Fake precision at frozen threshold | 97.90% | 94.81–99.81% |
| Fake recall at frozen threshold | 94.48% | 86.81–99.76% |

The detector flagged 2,653 fake recordings, missed 155, correctly accepted 2,959 real recordings and falsely flagged 57. EER describes the ROC crossing and did not set the deployed threshold. Intervals resample ten whole speaker folders with replacement 500 times. They describe uncertainty within this corpus and are approximate with so few groups. Four test folders contain only one class. Speaker-29 alone accounts for 137 of 155 misses and 45 of 57 false positives. [Complete per-speaker metrics](pipeline/outputs/audited_v1/full/per_speaker.csv)

The new AUC is numerically 0.0309 above 0.963. The evaluation unit changed from historical feature rows to whole recordings. The splits, training examples, filtering and selection procedure also changed. There is no defensible numerical estimate of improvement over the historical detector from these two figures.

## External app behavior (development only)

The frozen rule ran on all 44 recordings. An alert episode starts whenever the running score crosses into FAKE, including on the first eligible chunk. The final verdict is the state at file end. Timing uses elapsed audio time from the start, with the last partial chunk ending at its true duration.

| Duration | Fake at file end | Fake ever alerted | Genuine files | Genuine hours | False alert episodes | Median first alert among alerted fakes |
|---|---:|---:|---:|---:|---:|---:|
| Under 30 s | 3/3 | 3/3 | 14 | 0.024 | 0 | 1 s |
| 30 to under 120 s | 0/6 | 1/6 | 2 | 0.048 | 0 | 1 s |
| 120 s or longer | 3/13 | 7/13 | 6 | 0.992 | 0 | 1 s |
| All | **6/22** | **11/22** | **22** | **1.065** | **0** | **1 s** |

Eleven fake files never alerted. Five alerted temporarily and ended REAL. We observed **0 false alert episodes per genuine recording hour**, also zero over **0.843 hours of estimated active audio** (20 ms frames with RMS ≥ −45 dBFS). Speech-only hours require annotations. The rate is based on little exposure, and six long real files dominate it. Duration and source content vary together, so these bins cannot isolate a causal length effect. First-alert timing excludes misses and measures audio position, not hardware latency or delay from the onset of a manipulated segment. Live microphone capture itself was not exercised.

## Artifacts and reproduction

[New chunk manifest](pipeline/outputs/audited_v1/manifest.csv) · [Historical row crosswalk](pipeline/outputs/audited_v1/historical_manifest.csv) · [Duplicate source evidence](pipeline/outputs/audited_v1/historical_duplicates.csv) · [Audit and undecodable paths](pipeline/outputs/audited_v1/audit.json)

[Frozen checkpoint](pipeline/outputs/audited_v1/full/checkpoint.pth) · [Decision rule and hashes](pipeline/outputs/audited_v1/full/rule.json) · [Run configuration](pipeline/outputs/audited_v1/full/config.json) · [Installed packages](pipeline/outputs/audited_v1/full/environment.txt) · [Reproduction commands](pipeline/README.md)

[Recording predictions and source row IDs](pipeline/outputs/audited_v1/full/test_predictions.csv) · [Chunk scores](pipeline/outputs/audited_v1/full/test_chunks.csv) · [External file verdicts and first alerts](pipeline/outputs/audited_v1/full/external_predictions.csv) · [External duration bins](pipeline/outputs/audited_v1/full/external_bins.csv)

The versioned run is local under the repository's ignored outputs directory. Keep it with the raw data when moving machines. The recorded test speakers have now been evaluated. Future decisions informed by these results need fresh confirmation speakers.
