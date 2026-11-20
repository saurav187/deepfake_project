"""Train on manifest train rows, freeze on dev, then score the held-out recordings once."""
import argparse
from collections import Counter
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, recall_score,
                             confusion_matrix)
from torch.utils.data import DataLoader, TensorDataset

from audio_features import ROOT, RUN, SR, feature, read_audio
from model import CNNLSTMDetector
from prepare_manifest import digest, write_csv
from decision import trajectory, summarize


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def read_manifest(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r['row_id']) for r in rows] == list(range(len(rows)))
    active = [r for r in rows if r['split'] != 'excluded']
    for key in ['speaker_group','source_file','feature_hash','pcm_hash','legacy_hash']:
        owners = {}
        for r in active:
            value = r[key]
            if value:
                assert owners.setdefault(value,r['split']) == r['split'], (key,value)
    return active


def group_files(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['source_file'], []).append(r)
    return groups


def make_loader(rows, shuffle, seed):
    def load_record(items):
        audio = read_audio(ROOT/items[0]['source_file'])
        values = []
        for row in items:
            start = int(row['chunk'])*SR
            part = audio[start:start+SR]
            x = feature(part)
            assert len(part)==int(row['samples']) and digest(x)==row['feature_hash'], row
            values.append((row,x))
        return values
    # Fill a single in-memory array without copying raw recordings to disk.
    X = torch.empty(len(rows),1,128,94)
    y = torch.empty(len(rows))
    ids = torch.empty(len(rows),dtype=torch.long)
    with ThreadPoolExecutor(4) as pool:
        pos = 0
        for items in pool.map(load_record,group_files(rows).values(),buffersize=12):
            for row,x in items:
                X[pos,0] = torch.from_numpy(x)
                y[pos] = int(row['label'])
                ids[pos] = int(row['row_id'])
                pos += 1
    return DataLoader(TensorDataset(X,y,ids), batch_size=96, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def verify_ids(seen, rows):
    assert sorted(seen)==sorted(int(r['row_id']) for r in rows), 'Loader row mismatch'


@torch.inference_mode()
def predict(model,loader,rows,device):
    model.eval()
    scores = {}
    seen = []
    for x,_,ids in loader:
        values = model(x.to(device)).flatten().sigmoid().cpu().tolist()
        scores.update(zip(ids.tolist(),values))
        seen.extend(ids.tolist())
    verify_ids(seen,rows)
    return scores


def records(rows,scores,rule):
    result=[]
    for name,items in group_files(rows).items():
        items=sorted(items,key=lambda r:int(r['chunk']))
        ids=[int(r['row_id']) for r in items]
        ends=[int(r['chunk'])+int(r['samples'])/SR for r in items]
        result.append(dict(source_file=name,speaker_group=items[0]['speaker_group'],
                           label=int(items[0]['label']),row_ids=' '.join(map(str,ids)),
                           **summarize([scores[i] for i in ids],ends,rule)))
    return result


def metrics(items):
    y=[r['label'] for r in items]
    s=[r['score'] for r in items]
    pred=[r['verdict']=='FAKE' for r in items]
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    auc=eer=None
    if len(set(y))==2:
        auc=float(roc_auc_score(y,s))
        fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
        d=fpr+tpr-1
        j=int(np.searchsorted(d,0))
        eer=float(np.interp(0,d[max(0,j-1):j+1],fpr[max(0,j-1):j+1]))
    return dict(n=len(y),auc=auc,eer=eer,precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,tn=tn,fp=fp,fn=fn,tp=tp)


def fit_rule(rows,scores):
    best=-np.inf
    for agg in ['mean','fraction']:
        rule={'aggregation':agg,'threshold':0.5}
        items=records(rows,scores,rule)
        f,t,thresholds=roc_curve([r['label'] for r in items],[r['score'] for r in items])
        valid=np.isfinite(thresholds)
        idx=int(np.argmax(np.where(valid,t-f,-np.inf)))
        if (t-f)[idx]>best:
            best=(t-f)[idx]
            chosen=dict(aggregation=agg,threshold=float(thresholds[idx]))
    return chosen


def run(args):
    out=args.out
    out.mkdir(exist_ok=False,parents=True)
    seed=42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    rows=read_manifest(args.manifest)
    splits={k:[r for r in rows if r['split']==k] for k in ['train','dev','test']}
    if args.smoke:
        # Smoke evaluation uses three disjoint training speaker sets only.
        groups=sorted({r['speaker_group'] for r in splits['train']})
        partitions=np.array_split(groups,3)
        splits={k:[] for k in splits}
        for key,part in zip(splits,partitions):
            for label in ['0','1']:
                files=group_files([r for r in rows if r['speaker_group'] in part and r['label']==label])
                for name in list(files)[:12]:
                    splits[key].extend(files[name])
    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','librosa','soundfile','matplotlib','soxr','numba']}
    config=dict(seed=seed,epochs=args.epochs,smoke=args.smoke,python=platform.python_version(),
                versions=versions,device=str(device),cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                manifest_sha256=sha(args.manifest),batch_size=96,lr=.0005,
                code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    train,dev,test=(splits[k] for k in ['train','dev','test'])
    loaders=[make_loader(rows,shuffle,seed) for rows,shuffle in [(train,True),(dev,False)]]
    model=CNNLSTMDetector().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    c=Counter(r['label'] for r in train)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([c['0']/c['1']],device=device))
    best=-1
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        seen=[]
        for x,y,ids in loaders[0]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
                loss=criterion(model(x.to(device)).flatten(),y.to(device))
            loss.backward()
            opt.step()
            seen.extend(ids.tolist())
        verify_ids(seen,train)
        scores=predict(model,loaders[1],dev,device)
        items=records(dev,scores,dict(aggregation='mean',threshold=.5))
        auc=metrics(items)['auc']
        history.append(dict(epoch=epoch,dev_auc=auc))
        print(history[-1],flush=True)
        if auc>best:
            best=auc
            rule=fit_rule(dev,scores)
            rule.update(epoch=epoch,dev_auc=auc,version=1)
            torch.save(model.state_dict(),out/'checkpoint.pth')
    del loaders
    write_csv(out/'history.csv',history)
    rule['checkpoint_sha256']=sha(out/'checkpoint.pth')
    rule['manifest_sha256']=config['manifest_sha256']
    (out/'rule.json').write_text(json.dumps(rule,indent=2))
    # No test features or scores have been loaded before this point.
    model.load_state_dict(torch.load(out/'checkpoint.pth',weights_only=True,map_location=device))
    scores=predict(model,make_loader(test,False,seed),test,device)
    items=records(test,scores,rule)
    write_csv(out/'test_predictions.csv',items)
    write_csv(out/'test_chunks.csv',[dict(row_id=i,score=s) for i,s in scores.items()])
    groups=sorted({r['speaker_group'] for r in items})
    by_group=[[r for r in items if r['speaker_group']==g] for g in groups]
    write_csv(out/'per_speaker.csv',[dict(speaker_group=g,**metrics(v)) for g,v in zip(groups,by_group)])
    rng=np.random.default_rng(seed)
    boot=[metrics([r for j in rng.integers(len(groups),size=len(groups)) for r in by_group[j]])
          for _ in range(500)]
    intervals={}
    for k in ['auc','eer','precision','recall']:
        values=[m[k] for m in boot if m[k] is not None]
        intervals[k]=np.percentile(values,[2.5,97.5]).tolist() if values else None
    result=dict(metrics=metrics(items),speaker_bootstrap_95=intervals)
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots()
    ax.boxplot([[r['score'] for r in items if r['label']==y] for y in [0,1]],tick_labels=['Real','Fake'])
    ax.axhline(rule['threshold'],color='red'); ax.set_ylabel('Recording score')
    fig.savefig(out/'test_scores.png'); plt.close(fig)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=RUN/'manifest.csv')
    p.add_argument('--out',type=Path,default=RUN/'full')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
