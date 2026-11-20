"""Regression checks. The mock test split and replay fixture use training data only."""
import argparse
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from audio_features import ROOT, RUN, SR, read_audio

train = importlib.import_module('02_train_cnn_lstm')
inference = importlib.import_module('04_realtime_inference')


class BaselineTests(unittest.TestCase):
    def test_noncontiguous_rows(self):
        rows=train.read_manifest(self.manifest)
        selected=[row for row in rows if row['split']=='train'][4:85:13][::-1]
        loader=train.make_loader(selected,False,42)
        actual=[i for _,_,ids in loader for i in ids.tolist()]
        train.verify_ids(actual,selected)
        with self.assertRaises(AssertionError):
            train.verify_ids(list(range(len(selected))),selected)

    def test_replay(self):
        rows=train.read_manifest(self.manifest)
        name=next(r['source_file'] for r in rows if r['split']=='train' and int(r['chunk'])>2)
        audio=read_audio(ROOT/name)[:2*SR+731]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'replay.wav'
            sf.write(path,audio,SR,subtype='FLOAT')
            detector=inference.Detector(self.checkpoint,'cpu')
            file_result=inference.process_file(detector,path)
            blocks=(audio[i:i+317] for i in range(0,len(audio),317))
            live=inference.microphone(detector,blocks)
            self.assertEqual(file_result.keys(),live.keys())
            for key in live:
                if isinstance(live[key],float):
                    self.assertAlmostEqual(live[key],file_result[key],places=5)
                else:
                    self.assertEqual(live[key],file_result[key])
            self.assertEqual(live['duration_s'],len(audio)/SR)
            silent=inference.microphone(detector,[np.zeros(SR,dtype='float32')])
            self.assertEqual(silent['verdict'],'INSUFFICIENT')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=RUN/'manifest.csv')
    p.add_argument('--checkpoint',type=Path,default=RUN/'full/checkpoint.pth')
    args=p.parse_args()
    BaselineTests.manifest=args.manifest
    BaselineTests.checkpoint=args.checkpoint
    torch.set_num_threads(1)
    unittest.main(argv=['test_baseline'])
