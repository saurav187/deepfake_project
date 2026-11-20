"""File, folder and microphone inference using the frozen baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from audio_features import RUN, SR, feature, read_audio, windows
from decision import summarize
from model import CNNLSTMDetector


class Detector:
    def __init__(self, checkpoint=RUN/'full/checkpoint.pth', device=None):
        checkpoint=Path(checkpoint)
        self.rule=json.loads(checkpoint.with_name('rule.json').read_text())
        with checkpoint.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=self.rule['checkpoint_sha256']:
                raise ValueError('Checkpoint differs from frozen rule')
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model=CNNLSTMDetector().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint,map_location=self.device,weights_only=True))
        self.model.eval()

    @torch.inference_mode()
    def scores(self, features):
        tensor=torch.from_numpy(np.stack(features)[:,None]).to(self.device)
        return self.model(tensor).flatten().sigmoid().cpu().tolist()


class Stream:
    def __init__(self, detector):
        self.detector=detector
        self.tail=np.empty(0,dtype=np.float32)
        self.scores=[]
        self.ends=[]
        self.samples=0

    def _consume(self, audio):
        batch=[]
        for chunk in windows(audio):
            self.samples+=len(chunk)
            x=feature(chunk)
            if np.ptp(x)>0:
                batch.append(x)
                self.ends.append(self.samples/SR)
        if batch:
            self.scores.extend(self.detector.scores(batch))

    def feed(self, block):
        data=np.concatenate((self.tail,block))
        end=(len(data)//SR)*SR
        self._consume(data[:end])
        self.tail=data[end:].copy()

    def status(self):
        return summarize(self.scores,self.ends,self.detector.rule)

    def finish(self):
        self._consume(self.tail)
        self.tail=np.empty(0,dtype=np.float32)
        return dict(duration_s=self.samples/SR,**self.status())


def process_file(detector,path):
    stream=Stream(detector)
    audio=read_audio(path)
    for start in range(0,len(audio),32*SR):
        stream.feed(audio[start:start+32*SR])
    return stream.finish()


def microphone(detector,blocks,on_update=None):
    stream=Stream(detector)
    for block in blocks:
        before=len(stream.scores)
        stream.feed(block)
        if on_update and len(stream.scores)>before:
            on_update(stream.status())
    return stream.finish()


def mic_blocks(duration):
    import sounddevice as sd
    with sd.InputStream(samplerate=SR,channels=1,dtype='float32') as mic:
        remaining=round(duration*SR)
        while remaining>0:
            size=min(1600,remaining)
            data,overflow=mic.read(size)
            if overflow:
                raise RuntimeError('Microphone input overflow')
            yield data[:,0]
            remaining-=size


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['file','folder','mic'],default='file')
    p.add_argument('--checkpoint',type=Path,default=RUN/'full/checkpoint.pth')
    p.add_argument('--input_file',type=Path)
    p.add_argument('--input_dir',type=Path)
    p.add_argument('--output_csv',type=Path)
    p.add_argument('--duration',type=float,default=30)
    p.add_argument('--quiet',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    detector=Detector(args.checkpoint)
    if args.mode=='mic':
        print(microphone(detector,mic_blocks(args.duration),None if args.quiet else print))
    elif args.mode=='file':
        print(process_file(detector,args.input_file))
    else:
        if not args.output_csv or not args.input_dir:
            p.error('folder mode requires --input_dir and --output_csv')
        paths=sorted(p for p in args.input_dir.rglob('*') if p.suffix.lower() in {'.wav','.flac','.ogg','.m4a','.mp3'})
        with args.output_csv.open('x',newline='') as f:
            for i,path in enumerate(paths):
                row=dict(file=str(path.relative_to(args.input_dir)),**process_file(detector,path))
                if i==0:
                    writer=csv.DictWriter(f,fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)


if __name__=='__main__':
    main()
