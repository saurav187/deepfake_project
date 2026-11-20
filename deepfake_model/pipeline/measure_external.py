"""Measure recurring alert episodes and end-of-recording labels on the VA development data."""
import argparse
import importlib
from pathlib import Path
import numpy as np
import torch

from audio_features import BASE,RUN,SR,read_audio
from prepare_manifest import write_csv

inference=importlib.import_module('04_realtime_inference')


def active_seconds(path):
    # An explicit activity proxy; without transcripts or VAD annotations,
    # speech-only hours cannot be measured exactly.
    audio=read_audio(path)
    frame=SR//50
    padded=np.pad(audio,(0,(-len(audio))%frame))
    rms=np.sqrt(np.mean(np.square(padded.reshape(-1,frame)),axis=1))
    active=rms >= 10**(-45/20)
    lengths=np.minimum(frame,len(audio)-np.arange(len(active))*frame)
    return float(lengths[active].sum()/SR)


def bucket(items):
    fake=[r for r in items if r['label']==1]
    real=[r for r in items if r['label']==0]
    hours=sum(r['duration_s'] for r in real)/3600
    times=[r['first_alert_s'] for r in fake if r['first_alert_s'] is not None]
    alerts=sum(r['alerts'] for r in real)
    active_hours=sum(r['active_s'] for r in real)/3600
    return dict(fake=len(fake),fake_final=sum(r['verdict']=='FAKE' for r in fake),
                fake_ever=len(times),real=len(real),real_final_fake=sum(r['verdict']=='FAKE' for r in real),
                false_alert_episodes=alerts,real_hours=hours,alerts_per_hour=alerts/hours if hours else None,
                active_real_hours=active_hours,alerts_per_active_hour=alerts/active_hours if active_hours else None,
                median_first_alert_s=float(np.median(times)) if times else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=RUN/'full')
    args=parser.parse_args()
    torch.set_num_threads(2)
    detector=inference.Detector(args.run/'checkpoint.pth')
    root=BASE/'datasets'/'va_dataset'
    rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        rows.append(dict(source_file=str(path.relative_to(root)),label={'real':0,'fake':1}[path.parent.name],
                         active_s=active_seconds(path) if path.parent.name=='real' else None,
                         **inference.process_file(detector,path)))
    assert len(rows)==44
    write_csv(args.run/'external_predictions.csv',rows)
    summary=[]
    for name,lo,hi in [('all',0,np.inf),('<30s',0,30),('30-119s',30,120),('120s+',120,np.inf)]:
        summary.append(dict(duration_bin=name,**bucket([r for r in rows if lo<=r['duration_s']<hi])))
    write_csv(args.run/'external_bins.csv',summary)
    print(summary)


if __name__=='__main__':
    main()
