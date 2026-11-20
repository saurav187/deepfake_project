"""Crosswalk the historical arrays to the regenerated, source-addressed corpus."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from audio_features import RUN
from prepare_manifest import digest,write_csv


def crosswalk(out, data):
    with (out/'manifest.csv').open() as f:
        files={}
        for row in csv.DictReader(f):
            source=row['source_file']
            if source not in files or row['split']!='excluded':
                files[source]=row
    by_key=defaultdict(list)
    for name,row in files.items():
        by_key[row['legacy_hash'],row['speaker_group'],int(row['label'])].append(name)
    old=np.load(data/'X_features.npy',mmap_mode='r')
    labels=np.load(data/'y_labels.npy')
    speakers=np.load(data/'speaker_ids.npy')
    names={v:k for k,v in json.loads((data/'speaker_map.json').read_text()).items()}
    offsets=defaultdict(int)
    rows=[]
    for i,x in enumerate(old):
        key=(digest(x),names[int(speakers[i])],int(labels[i]))
        candidates=sorted(by_key[key])
        j=offsets[key]
        name=candidates[j]
        offsets[key]+=1
        row=files[name]
        rows.append(dict(row_id=i,source_file=name,speaker_group=key[1],label=key[2],
                         split=row['split'],fold=row['fold'],feature_hash=key[0],
                         matching_sources=len(candidates),source_assignment='unique' if len(candidates)==1
                         else 'interchangeable within hash/group/label',reason=row['reason']))
    assert all(n==len(by_key[k]) for k,n in offsets.items())
    write_csv(out/'historical_manifest.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=RUN)
    p.add_argument('--data',type=Path,default=RUN.parents[1]/'data')
    args=p.parse_args()
    crosswalk(args.out,args.data)
