"""Measure how well the stored file format alone predicts IndieFake dev labels."""
import argparse
import csv
import importlib
from itertools import batched
from collections import Counter
from pathlib import Path

import torch
import soundfile as sf
from sklearn.metrics import roc_auc_score

from audio_features import ROOT, RUN, SR, feature, read_audio
from decision import summarize
from prepare_manifest import digest, write_csv


def matched_dev_scores(dev_files, output):
    # Retain exactly the non-excluded dev windows of the baseline manifest.
    included = {}
    with (RUN / 'manifest.csv').open() as source:
        for row in csv.DictReader(source):
            if row['split'] == 'dev' and row['source_file'] in dev_files:
                included.setdefault(row['source_file'], []).append(row)
    torch.set_num_threads(2)
    detector = importlib.import_module('04_realtime_inference').Detector(device='cpu')
    predictions = []
    for name, rows in included.items():
        audio = read_audio(ROOT / name)
        scores, ends = [], []
        for batch in batched(sorted(rows, key=lambda r: int(r['chunk'])), 64):
            features = []
            for row in batch:
                start = int(row['chunk']) * SR
                chunk = audio[start:start + SR]
                x = feature(chunk)
                assert digest(x) == row['feature_hash']
                features.append(x)
                ends.append((start + len(chunk)) / SR)
            scores.extend(detector.scores(features))
        predictions.append(dict(file=name, speaker=rows[0]['speaker_group'], label=int(rows[0]['label']),
                                **summarize(scores, ends, detector.rule)))
    write_csv(output, predictions)
    print('Frozen CNN–LSTM within 16 kHz mono dev files:',
          roc_auc_score([r['label'] for r in predictions], [r['score'] for r in predictions]),
          Counter((r['label'], r['verdict']) for r in predictions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New directory for the audit outputs")
    output = parser.parse_args().out
    output.mkdir(parents=True, exist_ok=False)
    with (RUN / 'manifest.csv').open() as source:
        files = {(r['source_file'], r['split'], int(r['label'])) for r in csv.DictReader(source)
                 if r['split'] in {'train', 'dev'}}
    counts = Counter()
    matched = set()
    for name, split, label in sorted(files):
        info = sf.info(ROOT / name)
        counts[split, (info.samplerate, info.channels), label] += 1
        if split == "dev" and (info.samplerate, info.channels) == (16000, 1):
            matched.add(name)

    labels, scores, weights = [], [], []
    rows = []
    for fmt in sorted({fmt for _, fmt, _ in counts}):
        train = [counts['train', fmt, y] for y in (0, 1)]
        dev = [counts['dev', fmt, y] for y in (0, 1)]
        # Give unseen dev formats the overall training prevalence.
        train_counts = train if sum(train) else [sum(n for (s, _, k), n in counts.items()
                                                     if s == 'train' and k == y) for y in (0, 1)]
        score = train_counts[1] / sum(train_counts)
        for label, count in enumerate(dev):
            if count:
                labels.append(label)
                scores.append(score)
                weights.append(count)
        rows.append([*fmt, *train, *dev, score])
    out = output / 'source_formats.csv'
    with out.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['native_sample_rate', 'channels', 'train_real', 'train_fake', 'dev_real',
                         'dev_fake', 'train_fraction_fake'])
        writer.writerows(rows)
    print('Dev recording AUC from source format alone:',
          roc_auc_score(labels, scores, sample_weight=weights))
    matched_dev_scores(matched, out.with_name("matched_format_dev.csv"))


if __name__ == '__main__':
    main()
