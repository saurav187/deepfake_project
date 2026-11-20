"""Regenerate source-addressed features and audit the anonymous historical arrays."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from scipy.spatial import cKDTree
from audio_features import ROOT, RUN, SR, feature, mel_power, read_audio, windows


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scan(path):
    try:
        audio = read_audio(path)
    except subprocess.CalledProcessError:
        return path, "", "", 0, 0., None, []
    legacy = None
    if len(audio) >= 400:
        db = np.clip(librosa.power_to_db(mel_power(audio), ref=np.max), -80, 0)[:, :94]
        legacy = np.pad(db, ((0,0),(0,94-db.shape[1])), constant_values=-80)
    chunks = []
    for offset, part in enumerate(windows(audio)):
        x = feature(part)
        chunks.append((offset, digest(x), float(np.ptp(x)) == 0))
    return (path, digest(audio), digest(audio[:15080]), len(audio),
            float(np.max(np.abs(audio), initial=0)), legacy, chunks)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    data = RUN.parents[1] / 'data'
    old = np.load(data/'X_features.npy', mmap_mode='r')[:,:,:,0]
    ids = np.load(data/'speaker_ids.npy')
    labels = np.load(data/'y_labels.npy')
    names = json.loads((data/'speaker_map.json').read_text())
    hashes = [digest(x) for x in old]
    sets = defaultdict(list)
    for i,h in enumerate(hashes):
        sets[h].append(i)
    # Search within the recorded label/group using a compact signature. Verify the
    # entire matrix before accepting a match across numerical library versions.
    trees = {}
    for key in set(zip(ids, labels)):
        indices = np.flatnonzero((ids==key[0]) & (labels==key[1]))
        trees[key] = (indices, cKDTree(old[indices, 2::12, 2::11].reshape(len(indices),-1)))
    sources = []
    matches = defaultdict(list)
    with ThreadPoolExecutor(5) as pool:
        for path, pcm, prefix, samples, peak, x, chunks in pool.map(scan, sorted(ROOT.rglob('*.wav')), buffersize=16):
            group, category = path.relative_to(ROOT).parts[:2]
            label = int(category.lower()=='deepfakes')
            rel = str(path.relative_to(ROOT))
            old_hash = ''
            if x is not None:
                indices, tree = trees[names[group],label]
                dist, j = tree.query(x[2::12,2::11].ravel())
                i = int(indices[j])
                if np.max(np.abs(old[i]-x)) >= 0.06:
                    candidates = tree.query_ball_point(x[2::12,2::11].ravel(), 0.7)
                    i = min((int(indices[k]) for k in candidates),
                            key=lambda k: np.max(np.abs(old[k]-x)), default=i)
                if np.max(np.abs(old[i]-x)) < 0.06:
                    old_hash = hashes[i]
                    matches[old_hash].append((rel, pcm, prefix, peak, group, label))
            sources.append((rel,group,label,pcm,samples,old_hash,chunks))
    duplicates = []
    for h, indices in sets.items():
        if len(indices)<2: continue
        paths = matches[h]
        for rel, pcm, prefix, peak, group, lab in paths:
            same_audio = sum(v[1]==pcm for v in paths)>1
            same_prefix = sum(v[2]==prefix for v in paths)>1
            cause = ('historical input entirely at -80 dB floor' if np.max(old[indices[0]]) == -80 else
                     'silent recording' if peak==0 else 'same decoded audio' if same_audio else
                     'same decoded prefix' if same_prefix else 'equal after feature extraction')
            duplicates.append(dict(legacy_hash=h,legacy_ids=' '.join(map(str,indices)),source_file=rel,
                                   speaker_group=group,label=lab,pcm_hash=pcm,prefix_hash=prefix,
                                   peak=peak,cause=cause))
    write_csv(out/'historical_duplicates.csv',duplicates)
    groups = sorted(set(s[1] for s in sources if s[1].startswith('Speaker-')))
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    partition = {g:('test' if i<10 else 'dev' if i<20 else 'train') for i,g in enumerate(groups)}
    owners = defaultdict(set)
    for rel,group,label,pcm,samples,h,chunks in sources:
        for key in [pcm,h]+[ch for _,ch,constant in chunks if not constant]:
            if key: owners[key].add((group,label))
    seen = set()
    rows = []
    for rel,group,label,pcm,samples,h,chunks in sources:
        reason = ''
        keys = [pcm,h]+[ch for _,ch,constant in chunks if not constant]
        if any(len(owners[k])>1 for k in keys if k): reason='shared between speakers or labels'
        elif group not in partition: reason='unknown speaker'
        elif pcm in seen or (h and h in seen): reason='duplicate recording or historical input'
        seen.update(k for k in [pcm,h] if k)
        for offset,ch,constant in chunks:
            why = reason or ('constant input' if constant else '')
            rows.append(dict(row_id=len(rows),source_file=rel,speaker_group=group,label=label,
                             split='excluded' if why else partition[group],fold='holdout_1',
                             chunk=offset,samples=min(SR,samples-offset*SR),feature_hash=ch,
                             pcm_hash=pcm,legacy_hash=h,reason=why))
    write_csv(out/'manifest.csv', rows)
    audit = dict(historical_rows=len(old), historical_duplicate_sets=sum(len(v)>1 for v in sets.values()),
                 matched_historical_sets=len(matches), total_historical_sets=len(sets),
                 legacy_duplicate_sets_resolved=sum(len(matches[h])==len(v) for h,v in sets.items() if len(v)>1),
                 source_files=len(sources),zero_length_sources=[r for r,_,_,_,n,_,_ in sources if not n],
                 feature_rows=len(rows), split_rows=dict(Counter(r['split'] for r in rows)))
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(audit)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=RUN)
    prepare(parser.parse_args().out)
